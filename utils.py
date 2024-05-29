from gymnasium.core import Wrapper


class ObservationFlattenerWrapper(Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs.flatten(), reward, terminated, truncated, info
    
    def reset(self):
        obs, info = self.env.reset()
        return obs.flatten(), info
