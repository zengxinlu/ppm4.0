import os

scenes = ['box', 'torus', 'clocks', 'conference', 'cornellbox', 'diamond', 'sibenik', 'sponza']


for sce in scenes:
    os.system('.\\SDK-build\\bin\\Release\\ProgressivePhotonMap.exe --model {sce} 0'.format(sce = sce))

# import os 

# for i in range(0, 90):
#     print(i)
#     os.system('.\\SDK-build\\bin\\Release\\ProgressivePhotonMap.exe --model box {num}'.format(num = i))
