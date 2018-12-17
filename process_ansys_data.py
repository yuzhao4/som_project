import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import SOM

ansys_data = pd.read_csv("./final sign (5) particle track file.csv",skiprows = 16,sep = '\t',
                         names = ['ParticleResidenceTime','ParticleID','ParticleXPosition','ParticleYPosition','ParticleZPosition' ,
                                  'ParticleXVelocity','ParticleYVelocity','ParticleZVelocity'])
features = ansys_data.shape[0]

zero_index = np.where(ansys_data['ParticleResidenceTime'] == 0)
zero_index = np.append(zero_index, features-1)

start = np.delete(zero_index,-1)
end = np.delete(zero_index,0)

features_per_particle = end - start
smallest_value = np.min(features_per_particle)

index = np.linspace(0, smallest_value, 5, endpoint = False,dtype=int)

chosen_frames = []

for i in index:
    indices = start + i
    first_image = ansys_data.loc[indices.tolist(),['ParticleXPosition','ParticleYPosition','ParticleZPosition']]
    second_image = ansys_data.loc[(indices+1).tolist(),['ParticleXPosition','ParticleYPosition','ParticleZPosition']]
    chosen_frames.append([first_image,second_image])


