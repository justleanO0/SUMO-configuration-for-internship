# SUMO-configuration-for-internship

Requirements:
Python 3.8.0
TensorFlow 2.4.0
Keras 2.4.0
SUMO 1.8.0

Installation guide on SUMO official website: [https://sumo.dlr.de/docs/Downloads.php](https://sumo.dlr.de/docs/Downloads.php)

Name the SUMO installation directory as "sumo-1.8.0"

Environment variables need to be added.

To train, simply run `clients4_converge.py`.

If the code starts to run without an error, it means the connection between SUMO and Python has been established correctly. The model trained in this code is a four-agent federated learning global model. For the internship, you should replace the model with your own model.

Pay attention to the "\gym_sumo\envs\sumo_env.py" file, especially the "SumoEnv(gym.Env)" class. It contains all necessary configurations of the road.

---------
If you encounter the error: "Error: Cannot re-register id: gym_sumo-v0",
execute the following code:
```
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if 'gym_sumo-v0' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]
```


If you encounter the error:"TraCIException: Connection 'default' is already active.", execute the following code:
```
import traci
traci.close()
```

It closes the connection between python and SUMO. Everytime when you restart the environment, remember to close the traci connection via "traci.close()" first. But you will lose the access to the road information from python due to the close of the environment.
