import os

scenes = ['box', 'torus', 'clocks', 'conference', 'cornellbox', 'diamond', 'sibenik', 'sponza']


for sce in scenes:
    os.system('.\\SDK-build\\bin\\Release\\ProgressivePhotonMap.exe --model {sce} 1'.format(sce = sce))