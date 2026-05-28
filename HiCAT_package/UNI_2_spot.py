import pandas as pd
import numpy as np
import scanpy as sc
import os
os.chdir('/Volumes/ES_mac/Hu_lab/Jing_project')

def patch_2_spot_weighted(spot_x_pixel,spot_y_pixel,patch_size_spot,feature_matrix, patch_size_hipt=16):
	spot_pixel_total = pd.DataFrame({"x_pixel":spot_x_pixel,"y_pixel":spot_y_pixel})
	total_area=patch_size_spot*patch_size_spot/(patch_size_hipt*patch_size_hipt)
	# Initialize the new feature matrix # latest method
	spot_feature_matrix = np.zeros((spot_pixel_total.shape[0], feature_matrix.shape[0]))
	# Iterate over each spot
	for i, (_, row) in enumerate(spot_pixel_total.iterrows()):
		x_center = row['x_pixel']
		y_center = row['y_pixel']
		# left upper corner
		location_left_upper_x = int((x_center-patch_size_spot/2)/ patch_size_hipt)
		percent_left_upper_x=((x_center-patch_size_spot/2)/ patch_size_hipt)%1
		location_left_upper_y = int((y_center-patch_size_spot/2)/ patch_size_hipt)
		percent_left_upper_y=((y_center-patch_size_spot/2)/ patch_size_hipt)%1
		feature_value_left_upper= feature_matrix [:,location_left_upper_x,location_left_upper_y]
		proportion_left_upper = (1-percent_left_upper_x)*(1-percent_left_upper_y)/total_area
		#right upper corner
		location_right_upper_x = int((x_center-patch_size_spot/2)/ patch_size_hipt)
		percent_right_upper_x=((x_center-patch_size_spot/2)/ patch_size_hipt)%1
		location_right_upper_y = int((y_center+patch_size_spot/2)/ patch_size_hipt)
		percent_right_upper_y=((y_center+patch_size_spot/2)/ patch_size_hipt)%1
		feature_value_right_upper= feature_matrix [:,location_right_upper_x,location_right_upper_y]
		proportion_right_upper = (1-percent_right_upper_x)*(percent_right_upper_y)/total_area
		#left lower corner
		location_left_lower_x= int((x_center+patch_size_spot/2)/ patch_size_hipt)
		percent_left_lower_x=((x_center+patch_size_spot/2)/ patch_size_hipt)%1
		location_left_lower_y= int((y_center-patch_size_spot/2)/ patch_size_hipt)
		percent_left_lower_y=((y_center-patch_size_spot/2)/ patch_size_hipt)%1
		feature_value_left_lower= feature_matrix [:,location_left_lower_x,location_left_lower_y]
		proportion_left_lower = (percent_left_lower_x)*(1-percent_left_lower_y)/total_area
		#right lower corner
		location_right_lower_x= int((x_center+patch_size_spot/2)/ patch_size_hipt)
		percent_right_lower_x=((x_center+patch_size_spot/2)/ patch_size_hipt)%1
		location_right_lower_y =int((y_center+patch_size_spot/2)/ patch_size_hipt)
		percent_right_lower_y=((y_center+patch_size_spot/2)/ patch_size_hipt)%1
		feature_value_right_lower= feature_matrix [:,location_right_lower_x,location_right_lower_y]
		proportion_right_lower = (percent_right_lower_x)*(percent_right_lower_y)/total_area
		#left rectangle
		number_left_rectangle_percent = 1-percent_left_upper_y
		feature_value_left_rectangle = np.sum(feature_matrix [:,location_left_upper_x+1:location_left_lower_x,location_left_upper_y],axis=(1))
		proportion_left_rectangle = number_left_rectangle_percent/total_area
	   #right rectangle
		number_right_rectangle_percent = percent_right_upper_y
		feature_value_right_rectangle = np.sum(feature_matrix [:,location_right_upper_x+1:location_right_lower_x,location_right_upper_y],axis=(1))
		proportion_right_rectangle = number_right_rectangle_percent/total_area
	   #upper rectangle
		number_upper_rectangle_percent = 1-percent_left_upper_x
		feature_value_upper_rectangle = np.sum(feature_matrix [:,location_left_upper_x,location_left_upper_y+1:location_right_upper_y],axis=(1))
		proportion_upper_rectangle = number_upper_rectangle_percent/total_area
	   #lower rectangle
		number_lower_rectangle_percent = percent_left_lower_x
		feature_value_lower_rectangle = np.sum(feature_matrix [:,location_left_lower_x,location_left_lower_y+1:location_right_lower_y],axis=(1))
		proportion_lower_rectangle = number_lower_rectangle_percent/total_area
		#main, or inside whole rectangles
		feature_value_main_rectangle = np.sum(feature_matrix[:,location_left_upper_x+1:location_left_lower_x, location_left_upper_y+1:location_right_upper_y],axis=(1,2))
		proportion_main_rectangle = 1/total_area
		proportion_list_total = [proportion_left_upper, proportion_right_upper, proportion_left_lower, proportion_right_lower, proportion_left_rectangle, proportion_right_rectangle, proportion_upper_rectangle, proportion_lower_rectangle, proportion_main_rectangle]
		value_list_total = [feature_value_left_upper, feature_value_right_upper, feature_value_left_lower, feature_value_right_lower, feature_value_left_rectangle, feature_value_right_rectangle, feature_value_upper_rectangle, feature_value_lower_rectangle, feature_value_main_rectangle]
		#Multiply each column of value_list_total by the row of proportion_list_total
		result_array = np.array(proportion_list_total)  * np.array(value_list_total).T
		# Sum across columns
		final_feature_value = result_array.sum(axis=1)
		spot_feature_matrix[i]=final_feature_value
	return spot_feature_matrix

########################################################################################################################

patch_size_spot=255 # Visium: 255 | Visium HD: 60

sample = 'female'
adata = sc.read_h5ad(f'./UNI_ExtractFeatures_V5/result/{sample}/uni_super_emb.h5ad')
adata.obs["x"] = adata.obsm["spatial"][:, 0] 
adata.obs["y"] = adata.obsm["spatial"][:, 1]
cols = adata.obs['x'].values.astype(int)
rows = adata.obs['y'].values.astype(int)
num_rows = rows.max() + 1  # 0-based indexing
num_cols = cols.max() + 1
X = adata.X
if hasattr(X, 'toarray'):
    X = X.toarray()
print(X.shape)

num_features = X.shape[1]
X_grid = np.zeros((num_rows, num_cols, num_features), dtype=X.dtype) # initialize feature matrix with 0

for i in range(X.shape[0]):
    r, c = rows[i], cols[i]
    X_grid[r, c, :] = X[i] # fill in the available patch-level image features

#--------------------------------------------------------------------------------------------------------------
### Previously, what I used to transpose the matrix was:
### combined_feature_matrix = np.transpose(X_grid, (2, 0, 1)) # transpose to (features, y, x)
combined_feature_matrix = np.transpose(X_grid, (2, 1, 0)) # transpose to (features, x, y)
print(combined_feature_matrix.shape)
#--------------------------------------------------------------------------------------------------------------

coords = sc.read_h5ad(f'./Tonsil_Visium_data/{sample}/{sample}.h5ad')
df = coords.obs.copy()

# get the pixel coordinates of the cropped image
buffer = 500
x_min = int(df['pixel_x'].min()) - buffer # x-origin of the cropped image
y_min = int(df['pixel_y'].min()) - buffer # y-origin of the cropped image
print(x_min,y_min)
pixel_x = (df['pixel_x'] - x_min).tolist()
pixel_y = (df['pixel_y'] - y_min).tolist()

tmp=patch_2_spot_weighted(spot_x_pixel=pixel_x,spot_y_pixel=pixel_y,patch_size_spot=patch_size_spot, feature_matrix=combined_feature_matrix, patch_size_hipt=16)
tmp=sc.AnnData(tmp)
tmp.obs=df
tmp.var.index=[str(i) for i in range(tmp.shape[1])]
tmp.write_h5ad(f"./Tonsil_Visium_data/{sample}/{sample}_UNI_V5_features.h5ad")
print()




