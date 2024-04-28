# SUMO-configuration-for-internship

Requirements:
Python 3.8.0
TensorFlow 2.4.0
Keras 2.4.0
SUMO 1.8.0

Installation guide on SUMO official website: [https://sumo.dlr.de/docs/Downloads.php](https://sumo.dlr.de/docs/Downloads.php)
Name the SUMO installation directory as "sumo-1.8.0"
Environment variables need to be added.

To train, simply run `test.py`.
To test, simply run `test_demo.py`.

If you encounter the error: "Error: Cannot re-register id: gym_sumo-v0",
execute the following code:
```
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if 'gym_sumo-v0' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]
```
