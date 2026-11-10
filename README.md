# DreamerV3 + Attention

Project is using DreamerV3 from TorchRL library. With attention mechanism to tell the model which observation in a state should be prioritised for its decision making

All training and evaluation runs are in runs, not pushed because size is too large 
The models (.pt files) are simply too big to be pushed to github, thus they will be shared via alternative means.

Best training runs are in experiment26, where I utilised continual learning through a curriculum. 

Best run thus far can drive in wet weather, handle traffic lights, vehicles and pedestrians. Fails in consistently making turns due to policy degradation

Documentation of commands that is runnable can be found in dreamerv3_attention/README.md

dreamerv3_attention/ is the folder where all my scripts are in

