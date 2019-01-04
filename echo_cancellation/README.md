# Echo cancellation network

Run echo cancellation network  
```
python EchoCancellation.py [-h] [--mode {ae, none}] [-b B] [--save SAVE]  
```
optional arguments:  
*--mode {ae,none}* ae - train autoencoder, none - run trained network  
*-b* batch size to use  
*--save* specify folder to save to (default: 'save') 