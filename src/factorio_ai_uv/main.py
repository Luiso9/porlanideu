"""
Main training script for Factorio Autonomous AI.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .config import (
    TRAINING_CONFIG,
    PATH_CONFIG,
    REWARD_WEIGHTS,
    OBS_SHAPE,
    CaptureConfig,
)
from .env import FactorioEnv
from .exceptions import (
    ConnectionError,
    ModNotFoundError,
    ModelError,
    DatasetError,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PATH_CONFIG.logs_dir / "training.log"),
    ],
)

logger = logging.getLogger(__name__)


class PretrainedModelLoader:
    """Load pre-trained model or initialize new one."""

    def __init__(self, model_path: Path = PATH_CONFIG.pretrained_model) -> None:
        self.model_path = model_path
        self._model: PPO | None = None

    def load_or_initialize(self, env: DummyVecEnv) -> PPO:
        """Load pre-trained model or initialize new one."""
        if self.model_path.exists():
            try:
                logger.info(f"Loading pre-trained model from {self.model_path}")
                model = PPO.load(str(self.model_path), env=env)
                logger.info("Pre-trained model loaded successfully")
                self._model = model
                return model
            except Exception as e:
                logger.warning(f"Failed to load pre-trained model: {e}")

        logger.info("Initializing new PPO model with random weights")
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=str(PATH_CONFIG.logs_dir),
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            policy_kwargs=dict(
                features_extractor_kwargs=dict(features_dim=128),
                net_arch=dict(pi=[64, 64], vf=[64, 64]),
            ),
        )
        self._model = model
        return model


class DatasetManager:
    """Manage experience dataset with persistence."""

    def __init__(self, path: Path = PATH_CONFIG.dataset, max_size: int = 20000) -> None:
        self.path = path
        self.max_size = max_size
        self._dataset: list[dict] = []

    def load(self) -> list[dict]:
        """Load dataset from disk."""
        if self.path.exists():
            try:
                import json
                with open(self.path, "r") as f:
                    data = json.load(f)
                self._dataset = data[-self.max_size:] if len(data) > self.max_size else data
                logger.info(f"Loaded {len(self._dataset)} transitions from dataset")
                return self._dataset
            except Exception as e:
                logger.warning(f"Could not load dataset: {e}")
        return []

    def save(self) -> None:
        """Save dataset to disk."""
        try:
            import json
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w") as f:
                json.dump(self._dataset, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save dataset: {e}")

    def add_transition(self, transition: dict) -> None:
        """Add transition to dataset."""
        self._dataset.append(transition)
        if len(self._dataset) > self.max_size:
            self._dataset = self._dataset[-self.max_size:]

    def get_dataset(self) -> list[dict]:
        """Get current dataset."""
        return self._dataset

    def __len__(self) -> int:
        return len(self._dataset)


def create_env(
    capture_config: CaptureConfig | None = None,
    rcon_host: str = "127.0.0.1",
    rcon_port: int = 27015,
    rcon_password: str = "factorio_ai_password",
) -> FactorioEnv:
    """Create Factorio environment."""
    from .env import FactorioEnv
    from .screen_capture import FactorioScreenCapture
    from .config import CaptureConfig
    
    if capture_config is None:
        capture_config = CaptureConfig()
    
    capture = FactorioScreenCapture(capture_config)
    env = FactorioEnv(
        capture=capture,
        rcon_host=rcon_host,
        rcon_port=rcon_port,
        rcon_password=rcon_password,
    )
    return env


def create_vec_env(env: FactorioEnv) -> DummyVecEnv:
    """Wrap environment for stable-baselines3."""
    env = Monitor(env, str(PATH_CONFIG.logs_dir))
    vec_env = DummyVecEnv([lambda: env])
    return vec_env


def train(
    train_steps_per_iter: int = TRAINING_CONFIG.train_steps_per_iter,
    n_iterations: int = TRAINING_CONFIG.n_iterations,
    capture_config: CaptureConfig | None = None,
    rcon_host: str = "127.0.0.1",
    rcon_port: int = 27015,
    rcon_password: str = "factorio_ai_password",
    verbose: bool = True,
) -> tuple[PPO, list[dict]]:
    """
    Main training loop.
    
    Args:
        train_steps_per_iter: Training steps per iteration
        n_iterations: Number of self-play iterations
        capture_config: Screen capture configuration
        verbose: Enable verbose logging
        
    Returns:
        (trained_model, dataset)
    """
    if verbose:
        logger.setLevel(logging.INFO)

    logger.info("=" * 60)
    logger.info("Factorio Autonomous AI - Kruise_Kontrol + Screen Vision + PPO")
    logger.info("=" * 60)

    # Initialize components
    logger.info("Initializing environment...")
    env = create_env(
        capture_config=capture_config,
        rcon_host=rcon_host,
        rcon_port=rcon_port,
        rcon_password=rcon_password,
    )
    
    if not env.connect():
        raise ConnectionError("Failed to connect to Factorio")

    vec_env = create_vec_env(env)
    logger.info("Environment wrapped for training")

    # Load or initialize model
    model_loader = PretrainedModelLoader()
    model = model_loader.load_or_initialize(vec_env)

    # Load dataset
    dataset_manager = DatasetManager()
    dataset = dataset_manager.load()
    PATH_CONFIG.dataset.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting training: {n_iterations} iterations, {train_steps_per_iter} steps/iter")
    logger.info(f"Initial dataset size: {len(dataset)}")

    try:
        for iteration in range(1, n_iterations + 1):
            iter_start = time.time()
            
            # Play one episode
            obs = vec_env.reset()
            episode_reward = 0.0
            steps = 0

            while steps < train_steps_per_iter:
                # Model selects action
                action, _ = model.predict(obs, deterministic=False)

                # Take step in environment
                obs, rewards, dones, infos = vec_env.step(action)

                # VecEnv returns arrays; take first element
                reward = float(rewards[0])
                done = bool(dones[0])
                info = infos[0] if infos else {}

                # Store transition
                transition = {
                    "observation": obs[0].transpose(1, 2, 0).tolist() 
                                 if hasattr(obs[0], 'transpose') else list(obs[0]),
                    "action": int(action[0]) if hasattr(action, '__len__') else int(action),
                    "reward": reward,
                    "done": done,
                    "info": info,
                }
                dataset.append(transition)

                # Keep dataset bounded
                if len(dataset) > 20000:
                    dataset = dataset[-20000:]

                episode_reward += reward
                steps += 1

                if done:
                    logger.info(
                        f"[Iter {iteration}] Episode finished after {steps} steps, "
                        f"total reward={episode_reward:.2f}, dataset size={len(dataset)}"
                    )
                    break

            # Retrain on accumulated dataset
            logger.info(f"[Iter {iteration}] Retraining PPO on {len(dataset)} transitions...")

            if len(dataset) >= 100:
                model.learn(
                    total_timesteps=train_steps_per_iter,
                    reset_num_timesteps=False,
                    tb_log_name="factorio_ppo",
                )

            # Save model and dataset
            PATH_CONFIG.model_dir.mkdir(parents=True, exist_ok=True)
            iter_model_path = PATH_CONFIG.model_dir / f"factorio_ai_iter_{iteration}"
            model.save(str(iter_model_path))
            dataset_manager = DatasetManager()
            dataset_manager._dataset = dataset
            dataset_manager.save()
            
            iter_time = time.time() - iter_start
            logger.info(
                f"  -> Model saved: {iter_model_path} "
                f"(iter time: {iter_time:.1f}s, dataset: {len(dataset)})"
            )

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")

    except Exception as e:
        logger.error(f"Training error: {e}")
        raise

    finally:
        # Final saves
        PATH_CONFIG.model_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(PATH_CONFIG.model_dir / "factorio_ai_final"))
        dataset_manager = DatasetManager()
        dataset_manager._dataset = dataset
        dataset_manager.save()
        
        logger.info("Training complete")
        logger.info(f"Final model: {PATH_CONFIG.model_dir / 'factorio_ai_final'}")
        logger.info(f"Dataset transitions: {len(dataset)}")

    return model, dataset


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Factorio Autonomous AI Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=TRAINING_CONFIG.train_steps_per_iter,
        help="Training steps per iteration"
    )
    parser.add_argument(
        "--iterations", 
        type=int, 
        default=TRAINING_CONFIG.n_iterations,
        help="Number of training iterations"
    )
    parser.add_argument(
        "--capture-width", 
        type=int, 
        default=84,
        help="Capture width"
    )
    parser.add_argument(
        "--capture-height", 
        type=int, 
        default=84,
        help="Capture height"
    )
    parser.add_argument(
        "--capture-interval", 
        type=int, 
        default=2,
        help="Capture every N ticks"
    )
    parser.add_argument(
        "--capture-region", 
        type=str,
        help="Capture region as 'top,left,width,height'"
    )
    parser.add_argument(
        "--rcon-host", 
        type=str, 
        default="127.0.0.1",
        help="RCON host"
    )
    parser.add_argument(
        "--rcon-port", 
        type=int, 
        default=27015,
        help="RCON port"
    )
    parser.add_argument(
        "--rcon-password", 
        type=str, 
        default="factorio_ai_password",
        help="RCON password"
    )
    parser.add_argument(
        "--max-episode-steps", 
        type=int, 
        default=5000,
        help="Max steps per episode"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Reduce logging verbosity"
    )
    parser.add_argument(
        "--test-only", 
        action="store_true",
        help="Only test environment, don't train"
    )

    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Build capture config
    from .config import CaptureConfig
    capture_config = CaptureConfig(
        size=(args.capture_width, args.capture_height),
        interval=args.capture_interval,
    )
    if args.capture_region:
        try:
            parts = [int(x.strip()) for x in args.capture_region.split(",")]
            if len(parts) == 4:
                capture_config.region = {
                    "top": parts[0],
                    "left": parts[1], 
                    "width": parts[2],
                    "height": parts[3],
                }
        except ValueError:
            logger.warning("Invalid capture region format, using auto-detect")

    if args.test_only:
        logger.info("Running environment test only...")
        from .env import FactorioEnv
        from .screen_capture import FactorioScreenCapture
        from .config import CaptureConfig as CC
        
        capture = FactorioScreenCapture(capture_config)
        env = FactorioEnv(capture=capture)
        
        if not env.connect():
            logger.error("Failed to connect to Factorio")
            return 1
            
        obs, _ = env.reset()
        logger.info(f"Initial obs shape: {obs.shape}")

        for action in [1, 5, 7, 8, 0]:
            obs, reward, done, truncated, info = env.step(action)
            logger.info(f"Action: {info.get('action')}, Reward: {reward:.2f}")
            time.sleep(1)

        env.close()
        logger.info("Test complete!")
        return 0

    # Run training
    try:
        # Create new CaptureConfig with RCON settings (frozen dataclass)
        capture_config = CaptureConfig(
            size=(capture_config.size[0], capture_config.size[1]),
            interval=capture_config.interval,
            region=capture_config.region,
        )
        
        model, dataset = train(
            train_steps_per_iter=args.steps,
            n_iterations=args.iterations,
            capture_config=capture_config,
            rcon_host=args.rcon_host,
            rcon_port=args.rcon_port,
            rcon_password=args.rcon_password,
        )
        logger.info("Training completed successfully")
        return 0
        
    except ConnectionError as e:
        logger.error(f"Connection failed: {e}")
        logger.error("Ensure Factorio is running with RCON enabled and Kruise_Kontrol_Remote mod loaded")
        return 1
    except ModNotFoundError as e:
        logger.error(f"Mod not found: {e}")
        logger.error("Install Kruise_Kontrol_Remote mod from https://github.com/Factorio-Access/Kruise_Kontrol_Remote")
        return 1
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())