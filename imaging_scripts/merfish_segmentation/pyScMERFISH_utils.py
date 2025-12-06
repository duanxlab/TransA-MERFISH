# from image processing, cell segmentation to clustering
# Lin Zhang lin.zhang@ucsf.edu linzhangtuesday@gmail.com
# Can Liu and Shaobo Zhang

from ClusterMap.clustermap import *
# this ipynb should be in the ClusterMap folder, so that the module is imported correctly. 
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from anndata import AnnData
import tifffile
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import scipy.io
import math
from tqdm import tqdm
from skimage.measure import label, regionprops
from scipy.sparse import issparse
import matplotlib as mpl
from skimage import morphology
from cellpose import core, utils, io, models, metrics, plot
from cellpose.io import logger_setup 
import read_roi
import matplotlib.patches as patches
import matplotlib.path as mpath
import time
import os
import pickle
from scipy.ndimage import center_of_mass
import glob
import h5py


def MIP(inputfilepath,sample_name,region_name):
      
     import numpy as np
     import tifffile as tiff
     import glob
     import os
    
     # Define the directory containing the DAPI images
     image_dir = inputfilepath+"images"
     
     # Get list of all z-plane images (sorted)
     import glob
     import os
     from natsort import natsorted

     file_list = natsorted(glob.glob(os.path.join(image_dir, "mosaic_DAPI*.tif")))
     #file_list = natsorted(glob.glob(os.path.join(inputfilepath, "*dapi_in_ROI*.tif")))


     # Load images into a stack
     image_stack = np.array([tiff.imread(f) for f in file_list])

     # Compute Maximum Intensity Projection (MIP)
     mip_image = np.max(image_stack, axis=0)

     # Define the output file path
     output_path = os.path.join(inputfilepath+sample_name+"_"+region_name+ "_dapi_in_ROI_MIP.tif")

     # Save the MIP as a new TIFF file
     tiff.imwrite(output_path, mip_image)

     print(f"Maximum Intensity Projection saved as {output_path}")


def Spots(dapipath,inputfilepath,sample_name,region_name,quantile,outputfilepath):
    spots = pd.read_csv(dapipath+'detected_transcripts.csv') 
    print(spots)
    
    # x and y are coordinates in the whole region section, numbers are in dot scales, unit is um
    temp = spots[['global_x', 'global_y']].values
    transcript_positions = np.ones((temp.shape[0], temp.shape[1]+1)) 
    transcript_positions[:, :-1] = temp 
    
    # Transform coordinates to mosaic pixel coordinates 
    transformation_matrix = pd.read_csv(dapipath + 'images/micron_to_mosaic_pixel_transform.csv', header=None, sep=' ').values
    transformed_positions = np.matmul(transformation_matrix, np.transpose(transcript_positions))[:-1] 
    print(transformed_positions)
    spots.loc[:, 'spot_location_1'] = [int(i) for i in transformed_positions[0, :]] 
    spots.loc[:, 'spot_location_2'] = [int(i) for i in transformed_positions[1, :]] 

    print("after micron_to_mosaic_pixel transformation")
    print(spots)
    
    print("adapt data format to ClusterMap")

    spots.rename(columns={'gene': 'gene_name'}, inplace=True)

    print("sort the gene name from A-Z, 0-314, e.g. 0 is Abtb2")
    gene_list = pd.DataFrame(spots['gene_name'].unique(), columns=['gene_name'])
    gene_list = gene_list.sort_values(by='gene_name')
    gene_list = gene_list.reset_index(drop=True)
    gene_list.to_csv(outputfilepath+'genelist.csv', header=False, index=False) 
    
    a1 = gene_list.iloc[:, 0].tolist() 

    print("add 1 to gene ID from A-Z, 1-315, e.g. 1 is Abtb2")
    gene_ID=list(map(lambda x: a1.index(x)+1, spots['gene_name'])) # 1 is Abtb2, 315 is mCherry

    #create a new column in spots, name is due to ClusterMap data input structure
    spots['gene']=gene_ID #  now the type is numpy.int64
    spots['gene']=spots['gene'].astype('int') # after astype('int'), the type is numpy.int32

    spots_min_information = spots[['gene_name','spot_location_1','spot_location_2','gene']].copy()

    print("concise table for transcripts (spots)")
    print(spots_min_information)

    #import matplotlib.path as mpath
    
    #ROI_name=sample_name+"_"+region_name+"_"+ROI_coordinate
    #print("ROI name is ",ROI_name)
    #roi_SC = read_roi.read_roi_file(inputfilepath + ROI_name+'.roi')

    # Get the polygon coordinates
    #x = roi_SC[ROI_name]['x']
    #y = roi_SC[ROI_name]['y']
    #ROI_polygon_points = list(zip(x, y))

    # Create a Path object
    #ROI_path = mpath.Path(ROI_polygon_points)

    #def is_inside_polygon(row):
    #    return ROI_path.contains_point((row['spot_location_1'], row['spot_location_2']))

    #spots_in_ROI = spots_min_information[spots_min_information.apply(is_inside_polygon, axis=1)]

    
    #print("Spots_in_ROI")
    #print(spots_in_ROI)

    #save this spots_in_ROI as a csv
    spots_min_information.to_csv(outputfilepath + 'spots_in_ROI.csv', index=False)  

    print("the format is compatible with ClusterMap now")


# inputs: spots_in_ROI, dapi_in_ROI_MIP
# Step 1: Splitting

def Tile_Filter_Split_ClusterMap(inputfilepath,sample_name,window_size,region_name,tilenum,outputfilepath):
    # Splitting tiles
    total_steps = 2  # for splitting
    progress_bar = tqdm(total=total_steps, desc="Overall Progress")

    print(inputfilepath+sample_name+'_'+region_name+'_dapi_in_ROI_MIP.tif')
    dapi_in_ROI_MIP = tifffile.imread(inputfilepath+sample_name+'_'+region_name+'_dapi_in_ROI_MIP.tif')
    spots_in_ROI = pd.read_csv(inputfilepath+'spots_in_ROI.csv')    
    
    img = dapi_in_ROI_MIP
    
    label_img = get_img(img, spots_in_ROI, window_size=window_size, margin=math.ceil(window_size*0.1))
    progress_bar.update(1)  

    out = split(img, label_img, spots_in_ROI, window_size=window_size, margin=math.ceil(window_size*0.1))
    progress_bar.update(1)  

    # Close the progress bar
    progress_bar.close()

    # Save the DataFrame using pickle, which can preserve the numpy
    out.to_pickle(outputfilepath + 'out.pkl')

    # Step 2: Plot an example tile
    # for 'spots', each tile has the information of all the spots information, however, the spot_location_1/2 coordinates are local
    tile_num = tilenum
    out.loc[tile_num,'spots']

    #plot all the spots and overlaped with dapi, which will be saved as .png later
    plt.figure(figsize=(8,8))

    plt.scatter(out.loc[tile_num,'spots']['spot_location_1'],out.loc[tile_num,'spots']['spot_location_2'],s=0.01)
    plt.savefig(outputfilepath+sample_name+region_name+"_Tile_"+str(tilenum)+".pdf")
    plt.savefig(outputfilepath+sample_name+region_name+"_Tile_"+str(tilenum)+".png")
    plt.show()

    # Step 3: Filter tiles with transcripts signals > 3,000 and Saving the tiles
    # get all the spot numbers of each tile into dot_num 
    dot_num=[]
    for i in range(len(out)):
        dot_num.append(out.loc[i,'spots'].shape[0])

    # note that filtering may also filter out those tiles that are not in square shape (at edges). 
    filtered_tile_list = [] # this is to contain all the index of those tiles with 3000+ dots (transcripts)
    for i in range(out.shape[0]):
        if out.loc[i,'spots'].shape[0] > 3000:
            filtered_tile_list.append(i)
    print('the number of tiles with 3000+ spots is', len(filtered_tile_list))

    print('their ID are:', filtered_tile_list) 

    # to plot the dapi image with filtered_tiles (only those in filtered_tile_list can be annotated)
    print('now display the location of the total', len(filtered_tile_list), 'filtered tiles out of total',len(out), 'splitted tiles')

    plt.figure(figsize=(20,20))
    plt.imshow(dapi_in_ROI_MIP,cmap='Greys') #colormap is greys
    plt.title('dapi with filtered tiles')
    x_ticks = np.arange(0, dapi_in_ROI_MIP.shape[1], window_size)
    y_ticks = np.arange(0, dapi_in_ROI_MIP.shape[0], window_size)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.grid(True)

    
    for i in range(len(x_ticks)):
        for j in range(len(y_ticks)):
            center_x = (x_ticks[i] + x_ticks[i + 1]) / 2 if i+1 != len(x_ticks) else (x_ticks[i] + dapi_in_ROI_MIP.shape[1]) / 2
            center_y = (y_ticks[j] + y_ticks[j + 1]) / 2 if j+1 != len(y_ticks) else (y_ticks[j] + dapi_in_ROI_MIP.shape[0]) / 2
            # else is the situation when the last one is smaller than window_size
            if i + j * len(x_ticks) in filtered_tile_list: #this is the only difference from previous tile display
                plt.text(center_x, center_y, f'{i + j * len(x_ticks)}', ha='center', va='center') 
    plt.savefig(outputfilepath+sample_name+region_name+"Dapi_FilteredTiles.pdf")
    plt.savefig(outputfilepath+sample_name+region_name+"Dapi_FilteredTiles.png")
    plt.show()

    # save the filtered tile information (including DAPI image, spots with spatial and gene informatio) into .PNG files within folder 'tiles'
    # the .png file is important for cellpose segmentation. If you use ClusterMap's own cell segmentation, it may take much longer time.

    import os
    # Create the tiles folder, skip if it already exists
    os.makedirs(outputfilepath+'tiles/', exist_ok=True)

    for tile_num in filtered_tile_list:
        fig=plt.figure(figsize=(10,10),frameon=False)
        plt.imshow(out.loc[tile_num,'img'],cmap='Greys',alpha=0.6) #here is to plot the dapi tile in greys
        plt.scatter(out.loc[tile_num,'spots']['spot_location_1'],out.loc[tile_num,'spots']['spot_location_2'],s=0.05) #plot all transcripts of this tile
        plt.axis('off')
        plt.tight_layout(pad=0, w_pad=0, h_pad=0) #Padding between the figure edge and the edges of subplots, as a fraction of the font size.
        filename=outputfilepath+'tiles/'+str(tile_num)+'.png' #note that there should already have a folder (manually created) called tiles.
        plt.savefig(filename, dpi=180, bbox_inches='tight', pad_inches=0)
        plt.close()
        print('now saving the tile as: ' + str(tile_num) +'.png')
        # it usually takes ~2 min to store 86 figures


# run cellpose, specify GPU=True

def Segmentation_by_Cellpose(inputfilepath,sample_name,region_name,window_size,modeldir,diameter,outputfilepath):
    logger_setup(); 
    files = io.get_image_files(inputfilepath + 'tiles','.png')
        
    images = [io.imread(f) for f in files]
    window_size=window_size
    modeldir= modeldir
    
    model = models.CellposeModel(gpu=True, pretrained_model=modeldir) 
    channels = [0,0] #it means grayscale image 

    print("Cellpose segmentation and saving npy files")
    masks, flows, styles = model.eval(images, diameter=diameter, channels=channels) # this is the speed-limiting part
    # each tile need 2-3s (GPU on), for 90 tiles is 3 min
    # take almost 0.5-5 hours hours on old mac with CPU

    # generate _seg.npy file (contains labeled mask information in a binary file format of NumPy) for each tile
    io.masks_flows_to_seg(images=images, masks=masks, flows=flows, file_names=files, diams=diameter,channels=channels) 

    # note that the cell mask number is not len(np.unique()) since value = 0 is not cell mask, but for those not belong to any cell
    #my_list=[images,masks,flows]
    #return(my_list)	
	

def Reshape_RemovalEdgeCell(inputfilepath,window_size):
    os.chdir(inputfilepath+'tiles/')
    for file in glob.glob("*_seg.npy"):
        tile_num = int(file.split('_')[0])  
        tile = np.load(inputfilepath+'tiles/'+file, allow_pickle=True).item()
        tile_mask_original = Image.fromarray(tile['masks'])
        tile_mask_reshaped = tile_mask_original.resize((int(window_size*1.2),int(window_size*1.2)))
        tile_mask_reshaped = np.array(tile_mask_reshaped)
        np.save(inputfilepath+'tiles/' + str(tile_num) + '_reshaped.npy', tile_mask_reshaped)
        print('now saving reshaped tile ID: ', str(tile_num),'_reshaped.npy')
    for file in glob.glob("*_reshaped.npy"):
        filename=os.path.join(inputfilepath+'tiles/',file)
        mask_in_tile = np.load(filename,allow_pickle=True)
        tile_num = int(file.split('_')[0])
        unique_cells = np.unique(mask_in_tile)
        edge_cells = []
        for cell_id in unique_cells:
            if cell_id == 0:  # Skip background
                continue
            cell_mask = mask_in_tile == cell_id
            on_top_edge = np.any(cell_mask[0:10, :])  # Top 10 rows
            on_bottom_edge = np.any(cell_mask[-10:, :])  # Bottom 10 rows
            on_left_edge = np.any(cell_mask[:, 0:10])  # Left 10 columns
            on_right_edge = np.any(cell_mask[:, -10:]) 

            # If the cell touches any edge, set those pixels to 0
            if on_top_edge or on_bottom_edge or on_left_edge or on_right_edge:
                mask_in_tile[cell_mask] = 0
                edge_cells.append(cell_id)
        np.save(inputfilepath+'tiles/' + str(tile_num) + '_reshaped_edge_cells_removal.npy', mask_in_tile)
        print('now saving reshaped_edge_cells_removal tile ID: ', str(tile_num),'_reshaped_edge_cells_removal.npy')
     # note that the cell mask number is not len(np.unique()) since value = 0 is not cell mask, but for those not belong to any cell

def stitch_tiles(old_tile, new_tile):
    #note the two inputs of this function should be the same size,
    #for old_tile (stitched_cell_masks), here the input is only the tile part e.g. [0:2400, 2400:4800] 2400x2400 size
    
    #print(len(np.unique(old_tile[old_tile>0])), 'is the old_tile cell mask number')
    #print(len(np.unique(new_tile[new_tile>0])), 'is the new_tile cell mask number')
    #print(np.max(old_tile)+1, 'is the largest cell id now')
    new_tile[new_tile != 0 ] += (np.max(old_tile)+300 if np.max(old_tile) > 0 else 0)

    #### find where the two overlaps and the cell label of them (make unique), remove the old ones
    overlap_mask = (old_tile > 0) & (new_tile > 0)
    overlap_cell_list_old = old_tile[overlap_mask]  
    unique_overlap_cells_old = np.unique(overlap_cell_list_old)    
    ##### directly remove those in old_tile       
    unique_cells_old = np.unique(old_tile[old_tile>0])
    unique_cells_new = np.unique(new_tile[new_tile>0])

    for cell_id in unique_cells_old:
        if cell_id in unique_overlap_cells_old:
           # print('now removing cell from old tile: ', cell_id)
            old_tile[old_tile == cell_id] = 0
    
    # Combine the labeled tiles
    combined_tile = old_tile + new_tile
    add_cell_number =   len(np.unique(combined_tile[combined_tile>0])) - len(unique_cells_old) 
    print('now stitched another', add_cell_number, 'cells')
    print('now the total combined_tile cell mask numbers now is', len(np.unique(combined_tile[combined_tile>0])))
    return combined_tile


def Reassign_Mask_Plotting_woROI(inputfilepath,window_size,sample_name,region_name):
    dapi = tifffile.imread(inputfilepath+'images/mosaic_DAPI_z0.tif')
    spots_in_ROI = pd.read_csv(inputfilepath+'spots_in_ROI.csv')
    dapi_in_ROI_MIP = tifffile.imread(inputfilepath+sample_name+'_'+region_name+'_dapi_in_ROI_MIP.tif')    
    #ROI_name=sample_name+"_"+region_name+"_"+ROI_coordinate
    
    #roi_SC = read_roi.read_roi_file(inputfilepath + sample_name+'_'+region_name+'_'+ROI_coordinate+'.roi')
    # Get the polygon coordinates
    #x = roi_SC[ROI_name]['x']
    #y = roi_SC[ROI_name]['y']
    #ROI_polygon_points = list(zip(x, y))
    #ROI_mask = Image.new('L', (dapi.shape[1], dapi.shape[0]), 0)
    #ImageDraw.Draw(ROI_mask).polygon(ROI_polygon_points, outline=1, fill=1) # create a mask with ROI area is 1, non-ROI is 0
    #ROI_mask_array = np.array(ROI_mask)
    
    dapi_in_ROI = dapi 
    out = pd.read_pickle(inputfilepath + 'out.pkl')

    label_img = get_img(dapi_in_ROI_MIP, spots_in_ROI, window_size=window_size, margin=math.ceil(window_size*0.1))

    stitched_cell_masks = np.zeros(label_img.shape, dtype=np.int16) 
    left = int(window_size*0.1)
    right= int(window_size*1.1) 
    os.chdir(inputfilepath+'tiles/')
    i=0
    for file in glob.glob("*_reshaped_edge_cells_removal.npy"):
        tile_mask_reshaped = np.load(inputfilepath+'tiles/'+file,allow_pickle=True)
        tile_num = int(file.split('_')[0])
        print('now is processing filtered tile #',tile_num)
        itemindex = np.where(label_img==tile_num) 
        index_x = itemindex[0][0]
        index_y = itemindex[1][0]
        
        if index_y == 0 :
            index_y = 200
        if index_x == 0 :
            index_x = 200
            
        stitched_cell_masks[index_x-left:index_x+right,index_y-left:index_y+right] = stitch_tiles(stitched_cell_masks[index_x-left:index_x+right,index_y-left:index_y+right], tile_mask_reshaped)
        i+=1
        print('Now total', i,'out of', len(glob.glob("*_reshaped_edge_cells_removal.npy")), 'filtered tiles got stitched.')
        print('----------')

    print("stitched_cell_masks -> reassigned_masks")
    # Label the image
    labeled_image = label(stitched_cell_masks) #?
    reassigned_masks = np.zeros_like(stitched_cell_masks)
    for prop in regionprops(labeled_image):
        coords = prop.coords
        for coord in coords:
            reassigned_masks[coord[0], coord[1]] = prop.label

    print(np.max(reassigned_masks), 'is the largest cell label')
    print(np.min(reassigned_masks), 'is the smallest cell label, i.e. no-cells')
    print(len(np.unique(reassigned_masks))-1, 'is the unique cell numbers')

    # adjust for 400
    reassigned_masks = reassigned_masks[200:-200, 200:-200]
    
    np.save(inputfilepath+'tiles/'  + 'reassigned_masks.npy', reassigned_masks)
    print('now saving all the cell masks as reassigned_masks.npy')
    print("reassigned_masks shape",reassigned_masks.shape)
    print("dapi shape",dapi.shape)

    print('plot reassigned_mask')
    plt.figure(figsize=(20,20))
    plt.imshow(reassigned_masks,cmap='tab20')
    x_ticks = np.arange(0, dapi_in_ROI.shape[1], window_size)
    y_ticks = np.arange(0, dapi_in_ROI.shape[0], window_size)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.grid(True)

    filtered_tile_list = [] # this is to contain all the index of those tiles with 3000+ dots (transcripts)
    for i in range(out.shape[0]):
        if out.loc[i,'spots'].shape[0] > 3000:
            filtered_tile_list.append(i)

    for i in range(len(x_ticks)):
        for j in range(len(y_ticks)):
            center_x = (x_ticks[i] + x_ticks[i + 1]) / 2 if i+1 != len(x_ticks) else (x_ticks[i] + dapi_in_ROI.shape[1]) / 2
            center_y = (y_ticks[j] + y_ticks[j + 1]) / 2 if j+1 != len(y_ticks) else (y_ticks[j] + dapi_in_ROI.shape[0]) / 2
            if i + j * len(x_ticks) in filtered_tile_list:
                plt.text(center_x, center_y, f'{i + j * len(x_ticks)}', ha='center', va='center')
    plt.savefig(inputfilepath + "Labeling_Mask_with_filtered_tiles.png")
    plt.savefig(inputfilepath + "Labeling_Mask_with_filtered_tiles.pdf")
    plt.axis('on')
    plt.show()
    return(reassigned_masks)
    # return stitched_cell_masks
# it usually takes 2 min for 86 tiles to be stitched.


### this is to transform the coordinates from Dots_scale (i.e. dots and ROIs use) to DAPI_Scale
def transform_coordinates_from_Dots_to_DAPI(x_coor_dots, y_coor_dots,inputfilepath):

    merged_array = np.column_stack((np.array(x_coor_dots), np.array(y_coor_dots)))
    
    transcript_positions = np.ones((merged_array.shape[0], merged_array.shape[1]+1)) #create a NumPy array(19008194x3) with all ones, temp.shape[0] means dimensions
    
    transcript_positions[:, :-1] = merged_array # array[:,:-1] means slicing the array from first column to last but except the last. 
  
    # Transform coordinates to mosaic pixel coordinates 
    transformation_matrix = pd.read_csv(inputfilepath + 'images/micron_to_mosaic_pixel_transform.csv', header=None, sep=' ').values
    
    transformed_positions = np.matmul(transformation_matrix, np.transpose(transcript_positions))[:-1] #this is matrix multiplication
    
    return transformed_positions
    
# transform DAPI scale back to Dots Scale
def transform_coordinates_from_DAPI_to_Dots(x_coor_dapi, y_coor_dapi,inputfilepath):
    # Read the transformation matrix
    transformation_matrix = pd.read_csv(inputfilepath + 'images/micron_to_mosaic_pixel_transform.csv', header=None, sep=' ').values
    
    # Compute the inverse of the transformation matrix
    inverse_transformation_matrix = np.linalg.inv(transformation_matrix)
    
    # Create an array with the transformed coordinates
    transformed_positions = np.array([x_coor_dapi, y_coor_dapi, 1])
    
    # Apply the inverse transformation
    original_positions = np.matmul(inverse_transformation_matrix, transformed_positions)
    
    # Extract the original x and y coordinates
    x_coor, y_coor = original_positions[:-1]
    
    return x_coor, y_coor


def ClusterMap_model(inputfilepath,sample_name,region_name):

    ## assign clustermap cell-id to each transcript using the filtered_masks id (-1 to max) # why minus 1 here
    spots_in_ROI = pd.read_csv(inputfilepath+'spots_in_ROI.csv')
    reassigned_masks = np.load(inputfilepath+'tiles/'  + 'reassigned_masks.npy')
    dapi_in_ROI_MIP = tifffile.imread(inputfilepath+sample_name+'_'+region_name+'_dapi_in_ROI_MIP.tif')    

    print("spots_in_ROI")
    print(spots_in_ROI)
    
    cell_ids = reassigned_masks[spots_in_ROI['spot_location_2'],spots_in_ROI['spot_location_1']] -1
    spots_in_ROI['clustermap'] = cell_ids 

    print("spots_in_ROI after adding clustermap ID")
    spots_in_ROI

    print(len(np.unique(reassigned_masks))-1, 'is the unique cell numbers with counts from reassigned_masks')
    print('max cell ID from clustermap')
    max(spots_in_ROI['clustermap'])

    # build ClusterMap model based on spots_in_ROI
    xy_radius=30 # how to define this?
    num_gene=np.max(spots_in_ROI['gene'])
    
    gene_list=np.arange(1,num_gene+1) 
    num_dims=len(dapi_in_ROI_MIP.shape)
    ClusterMap_model = ClusterMap(spots=spots_in_ROI, dapi=None, gene_list=gene_list, num_dims=num_dims,
                 xy_radius=xy_radius,z_radius=0,fast_preprocess=False)
    ClusterMap_model.plot_segmentation(figsize=(20,20),s=0.05,plot_with_dapi=False,plot_dapi=False, show=False)
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.savefig(inputfilepath + "ClusterMap_Model.png")
    plt.savefig(inputfilepath + "ClusterMap_Model.pdf")
    plt.show()    

    # use model.save_segmentation(filepath) to save the ClusterMap_model into a csv 
    path_save = inputfilepath+'ClutserMap-style_spots_with_cell_masks_by_cellpose_reassigned_masks_with200EachEdgeRemovedpostStitch_DebrisEdgeCellRemoved.csv'
    ClusterMap_model.save_segmentation(path_save) # how to read it back
    return(ClusterMap_model)

# create anndata with spatial info
def AnnData_setup(inputfilepath,adatafilepath,cell_number,sample_name,region_name):
    reassigned_masks = np.load(inputfilepath+'tiles/'  + 'reassigned_masks.npy')
    spots_in_ROI = pd.read_csv(inputfilepath+'spots_in_ROI.csv')
    dapi_in_ROI_MIP = tifffile.imread(inputfilepath+sample_name+'_'+region_name+'_dapi_in_ROI_MIP.tif')
    
    props = regionprops(label_image = reassigned_masks)
    ClusterMap_model.cellcenter = np.asarray([prop.centroid for prop in props])
    
    areas = [prop.area for prop in props]
    cellid='clustermap'
    geneid='gene'

    num_gene=np.max(spots_in_ROI['gene'])
    gene_list=np.arange(1,num_gene+1) 
    
    genes = pd.DataFrame(spots_in_ROI['gene_name'].unique(), columns=['gene_name'])
    genes = genes.sort_values(by='gene_name')
    genes = genes.reset_index(drop=True)
    genes.columns = [0]
    
    num_dims=len(dapi_in_ROI_MIP.shape)
    ClusterMap_model.create_cell_adata(cellid, geneid, gene_list, genes, num_dims)
    
    # note when create_cell_data, all those cells with no counts are removed
    ClusterMap_model.cell_adata      

    # Add obsm['spatial'] to adata
    ClusterMap_model.cell_adata.obsm["spatial"] = ClusterMap_model.cell_adata.obs[['row', 'col']].to_numpy()
    ClusterMap_model.cell_adata

    filtered_cell_num = len(ClusterMap_model.cell_adata.obsm['spatial'])
    filtered_cell_num

    # Convert the list to a 6141x1 array
    areas_array = np.array(areas).reshape(-1, 1)

    all_masks_spatial_information = np.hstack((ClusterMap_model.cellcenter, areas_array))

    x_corrdinates = ClusterMap_model.cell_adata.obsm['spatial'][:,0]

    # Create an empty array to store the filtered rows
    # read this part in detail later
    filtered_masks_spatial_information = np.empty((filtered_cell_num, 3))

    for i, x_coorrdinates in enumerate(x_corrdinates):
        for j, all_cells_information in enumerate(all_masks_spatial_information):
            if x_coorrdinates==all_cells_information[1]:
                filtered_masks_spatial_information[i] = all_cells_information
                break

    ClusterMap_model.cell_adata.obs['volume'] = filtered_masks_spatial_information[:,2]

    ClusterMap_model.cell_adata

    blank_genes_list = [name for name in ClusterMap_model.cell_adata.var_names if name.startswith('Blank-')]

    adata_blank_genes = ClusterMap_model.cell_adata[:, blank_genes_list]
    adata_blank_genes

    blank_genes_matrix = adata_blank_genes.X
    ClusterMap_model.cell_adata.obsm['blank_genes'] = blank_genes_matrix
    
    #filter the adata by deleting all the Blank genes in n_vars (now 300 instead of 315)
    non_blank_genes_list = [name for name in ClusterMap_model.cell_adata.var_names if not name.startswith('Blank-')]
    non_blank_genes_list
    ClusterMap_model.cell_adata = ClusterMap_model.cell_adata[:, non_blank_genes_list]

    ClusterMap_model.cell_adata.var_names

    print('the 238th gene is',ClusterMap_model.cell_adata.var_names[238])
    print('the 98th gene is',ClusterMap_model.cell_adata.var_names[98])

    print(ClusterMap_model.cell_adata.obs['volume'][cell_number], 'is the volume of the ', cell_number,' cell')
    print(ClusterMap_model.cell_adata.obsm['spatial'][cell_number], 'is the centroid location of the',cell_number,' cell')
    print(ClusterMap_model.cell_adata.X[cell_number, :].sum(), 'is the total counts of the', cell_number,' cell')
    print(ClusterMap_model.cell_adata.X[cell_number, :][238], 'is the total Slc17a6 counts of the ',cell_number,' cell')
    print(ClusterMap_model.cell_adata.X[cell_number, :][98], 'is the total Gad1 counts of the ', cell_number,' cell')

    adata = ClusterMap_model.cell_adata
    adata.var.index.name = 'genes'
    adata.write_h5ad(inputfilepath+"adata_in_SCROI_"+sample_name+"_"+region_name+".h5ad")
    adata.write_h5ad(adatafilepath+"adata_in_SCROI_"+sample_name+"_"+region_name+".h5ad")
    
    return(adata)


def plot_cell(inputfilepath, cell_ID_in_reassigned_masks,gene1, gene2):

    path_save = inputfilepath+'ClutserMapstyle_spots_with_cell_masks.csv'
    spots_in_ROI = pd.read_csv(path_save)
    spots_in_ROI
    
    reassigned_masks = np.load(inputfilepath+'tiles/'  + 'reassigned_masks.npy')
    
    # plot an example cell (cell_id is 'clustermap') 
    cellID_in_clustermap = cell_ID_in_reassigned_masks - 1
    filtered_df = spots_in_ROI[spots_in_ROI['clustermap'] == cellID_in_clustermap]
#   cell_ID_before_filter = inverted_dictionary.get(cell_ID)
    edge = 10
    y_indices, x_indices = np.where(reassigned_masks == cell_ID_in_reassigned_masks)
    cell_boundary_x_min = np.min(x_indices)-edge
    cell_boundary_x_max = np.max(x_indices)+edge
    cell_boundary_y_min = np.min(y_indices)-edge
    cell_boundary_y_max = np.max(y_indices)+edge
    
    plt.figure(figsize=(10, 10))
   
    # Crop the image
    cropped_image = reassigned_masks[cell_boundary_y_min:cell_boundary_y_max, cell_boundary_x_min:cell_boundary_x_max]
    plt.imshow(cropped_image, cmap='tab20', extent=[cell_boundary_x_min, cell_boundary_x_max, cell_boundary_y_max, cell_boundary_y_min])

    # get the cell center
    binary_mask = (cropped_image == cell_ID_in_reassigned_masks)
    labeled_mask = label(binary_mask)
    props = regionprops(labeled_mask)
    centroids = [prop.centroid for prop in props] #note the centroids is y,x 
    areas = [prop.area for prop in props]
    print('cell ID in reassigned_masks_new: ', cell_ID_in_reassigned_masks)
    print(' ')
    print("Centroid coordinates (shown as yellow dot) of this cell in DAPI scale: ", (int(centroids[0][1])+cell_boundary_x_min, int(centroids[0][0])+cell_boundary_y_min))
    print("Centroid coordinates (shown as yellow dot) of this cell in Dots scale in Visualizer: ", transform_coordinates_from_DAPI_to_Dots(int(centroids[0][1])+cell_boundary_x_min, int(centroids[0][0])+cell_boundary_y_min,inputfilepath=inputfilepath))
    print(' ')
    print("Volume of this cell: ", areas)
    print(' ')
    print('Total counts of this cell (white dots): ',len(filtered_df))


    blank_counts = 0
    for item in filtered_df['gene_name']:
        if "Blank-" in item:
            blank_counts += 1
    print(blank_counts, 'counts of blank genes')
    
    certain_spots1 = filtered_df[filtered_df['gene_name'] == gene1]
    print(gene1,'counts of this cell (green dots): ', len(certain_spots1))
    certain_spots2 = filtered_df[filtered_df['gene_name'] == gene2]
    print(gene2,'counts of this cell (green dots): ', len(certain_spots2))
    
    # Create a scatter plot of spot_location_1 vs. spot_location_2 for the filtered DataFrame
    plt.scatter(filtered_df['spot_location_1'], filtered_df['spot_location_2'], s = 5, color='white')
    plt.scatter(int(centroids[0][1])+cell_boundary_x_min, int(centroids[0][0])+cell_boundary_y_min, s = 100, color='yellow')
    plt.scatter(certain_spots1['spot_location_1'], certain_spots1['spot_location_2'], s = 20, color='green')
    plt.scatter(certain_spots2['spot_location_1'], certain_spots2['spot_location_2'], s = 30, color='orange')
    
    
    # Add labels and a title for clarity
    plt.xlabel('Spot Location 1')
    plt.ylabel('Spot Location 2')
    plt.title('Scatter Plot of Spot Locations of cell_ID_in reassigned_masks: '+ str(cell_ID_in_reassigned_masks))
    # plt.gca().invert_yaxis()
    # Display the plot
    plt.savefig(inputfilepath + "Cell_"+str(cell_ID_in_reassigned_masks)+"_"+"X_MERSCOPREVisual_Coordi_"+str(transform_coordinates_from_DAPI_to_Dots(int(centroids[0][1])+cell_boundary_x_min,int(centroids[0][0])+cell_boundary_y_min,inputfilepath=inputfilepath))+"_scatterPlot.png")
    plt.savefig(inputfilepath + "Cell_"+str(cell_ID_in_reassigned_masks)+"_"+"X_MERSCOPREVisual_Coordi_"+str(transform_coordinates_from_DAPI_to_Dots(int(centroids[0][1])+cell_boundary_x_min,int(centroids[0][0])+cell_boundary_y_min,inputfilepath=inputfilepath))+"_scatterPlot.pdf")

    plt.show()

# get cell_ID of reassgined_masks by coordinates
def get_cellID_from_coordinates(x_coordinate,y_coordinate,inputfilepath):
    reassigned_masks = np.load(inputfilepath+'tiles/'  + 'reassigned_masks.npy')
    cell_ID_reassigned_masks = reassigned_masks[y_coordinate][x_coordinate]
    return cell_ID_reassigned_masks

def get_cell_ID_in_adata_from_in_reassigned_masks(cell_ID_in_adata):
    return cell_IDs_in_reassigned_masks.index(cell_ID_in_adata)


def ClusterMap_model_and_Anndata(inputfilepath,adatafilepath,sample_name,region_name,cell_number,gene_1,gene_2,addmWmC):

    ## assign clustermap cell-id to each transcript using the filtered_masks id (-1 to max) # why minus 1 here
    spots_in_ROI = pd.read_csv(inputfilepath+'spots_in_ROI.csv')
    reassigned_masks = np.load(inputfilepath+'tiles/'  + 'reassigned_masks.npy')
    dapi_in_ROI_MIP = tifffile.imread(inputfilepath+sample_name+'_'+region_name+'_dapi_in_ROI_MIP.tif')

    print("spots_in_ROI")
    print(spots_in_ROI)
    
    cell_ids = reassigned_masks[spots_in_ROI['spot_location_2'],spots_in_ROI['spot_location_1']] -1
    spots_in_ROI['clustermap'] = cell_ids 

    print("spots_in_ROI after adding clustermap ID")
    spots_in_ROI

    print(len(np.unique(reassigned_masks))-1, 'is the unique cell numbers with counts from reassigned_masks')
    print('max cell ID from clustermap')
    max(spots_in_ROI['clustermap'])

    # build ClusterMap model based on spots_in_ROI
    xy_radius=30 # how to define this?
    num_gene=np.max(spots_in_ROI['gene'])
    
    gene_list=np.arange(1,num_gene+1) 
    num_dims=len(dapi_in_ROI_MIP.shape)
    ClusterMap_model = ClusterMap(spots=spots_in_ROI, dapi=None, gene_list=gene_list, num_dims=num_dims,
                 xy_radius=xy_radius,z_radius=0,fast_preprocess=False)
    ClusterMap_model.plot_segmentation(figsize=(20,20),s=0.05,plot_with_dapi=False,plot_dapi=False, show=False)
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.savefig(inputfilepath + "ClusterMap_Model.png")
    plt.savefig(inputfilepath + "ClusterMap_Model.pdf")
    plt.show()    

    # use model.save_segmentation(filepath) to save the ClusterMap_model into a csv 
    path_save = inputfilepath+'ClutserMapstyle_spots_with_cell_masks.csv'
    ClusterMap_model.save_segmentation(path_save) # how to read it back
    
    props = regionprops(label_image = reassigned_masks)
    ClusterMap_model.cellcenter = np.asarray([prop.centroid for prop in props])
    
    areas = [prop.area for prop in props]
    cellid='clustermap'
    geneid='gene'

    num_gene=np.max(spots_in_ROI['gene'])
    gene_list=np.arange(1,num_gene+1) 
    
    genes = pd.DataFrame(spots_in_ROI['gene_name'].unique(), columns=['gene_name'])
    genes = genes.sort_values(by='gene_name')
    genes = genes.reset_index(drop=True)
    genes.columns = [0]
    
    num_dims=len(dapi_in_ROI_MIP.shape)
    ClusterMap_model.create_cell_adata(cellid, geneid, gene_list, genes, num_dims)
    
    # note when create_cell_data, all those cells with no counts are removed
    ClusterMap_model.cell_adata      

    # Add obsm['spatial'] to adata
    ClusterMap_model.cell_adata.obsm["spatial"] = ClusterMap_model.cell_adata.obs[['row', 'col']].to_numpy()
    ClusterMap_model.cell_adata

    filtered_cell_num = len(ClusterMap_model.cell_adata.obsm['spatial'])
    filtered_cell_num

    # Convert the list to a 6141x1 array
    areas_array = np.array(areas).reshape(-1, 1)

    all_masks_spatial_information = np.hstack((ClusterMap_model.cellcenter, areas_array))

    x_corrdinates = ClusterMap_model.cell_adata.obsm['spatial'][:,0]

    # Create an empty array to store the filtered rows
    # read this part in detail later
    filtered_masks_spatial_information = np.empty((filtered_cell_num, 3))

    for i, x_coorrdinates in enumerate(x_corrdinates):
        for j, all_cells_information in enumerate(all_masks_spatial_information):
            if x_coorrdinates==all_cells_information[1]:
                filtered_masks_spatial_information[i] = all_cells_information
                break

    ClusterMap_model.cell_adata.obs['volume'] = filtered_masks_spatial_information[:,2]

    ClusterMap_model.cell_adata

    blank_genes_list = [name for name in ClusterMap_model.cell_adata.var_names if name.startswith('Blank-')]

    adata_blank_genes = ClusterMap_model.cell_adata[:, blank_genes_list]
    adata_blank_genes

    blank_genes_matrix = adata_blank_genes.X
    ClusterMap_model.cell_adata.obsm['blank_genes'] = blank_genes_matrix
    
    #filter the adata by deleting all the Blank genes in n_vars (now 300 instead of 315)
    non_blank_genes_list = [name for name in ClusterMap_model.cell_adata.var_names if not name.startswith('Blank-')]
    non_blank_genes_list
    ClusterMap_model.cell_adata = ClusterMap_model.cell_adata[:, non_blank_genes_list]

    ClusterMap_model.cell_adata.var_names

    genes=ClusterMap_model.cell_adata.var_names
    Slc17a6_index=genes.get_loc(gene_1)
    Gad1_index=genes.get_loc(gene_2)

    print('the '+ str(Slc17a6_index) +'th gene is',ClusterMap_model.cell_adata.var_names[Slc17a6_index])
    print('the '+ str(Gad1_index) +'th gene is',ClusterMap_model.cell_adata.var_names[Gad1_index])

    #print(ClusterMap_model.cell_adata.obs['volume'][cell_number], 'is the volume of the ', cell_number,' cell')
    #print(ClusterMap_model.cell_adata.obsm['spatial'][cell_number], 'is the centroid location of the',cell_number,' cell')
    #print(ClusterMap_model.cell_adata.X[cell_number, :].sum(), 'is the total counts of the', cell_number,' cell')
    #print(ClusterMap_model.cell_adata.X[cell_number, :][Slc17a6_index], 'is the total Slc17a6 counts of the ',cell_number,' cell')
    #print(ClusterMap_model.cell_adata.X[cell_number, :][Gad1_index], 'is the total Gad1 counts of the ', cell_number,' cell')

    if addmWmC:
        # Import mWmC+ ROIs from Vizgen Visualizer
        mWmC_cells_ROI = h5py.File(inputfilepath+sample_name+'_'+region_name+'_mWmC.hdf5', 'r') 

   	# Access a specific dataset
        data_X = mWmC_cells_ROI['obs']['center_x']
        data_Y = mWmC_cells_ROI['obs']['center_y']

    	# Add a obs['mWmC'] as True or False to anndata
        merged_array = np.column_stack((np.array(data_X), np.array(data_Y)))
        print('The orignal center coordinates of all mWmC ROIs: ',merged_array)
        transcript_positions = np.ones((merged_array.shape[0], merged_array.shape[1]+1)) #create a NumPy array(19008194x3) with all ones, temp.shape[0] means dimensions
        transcript_positions[:, :-1] = merged_array # array[:,:-1] means slicing the array from first column to last but except the last. 
        # here the transcript_positions all the third column is 1 (z)
        transformation_matrix = pd.read_csv(inputfilepath + 'images/micron_to_mosaic_pixel_transform.csv', header=None, sep=' ').values
        transformed_positions = np.matmul(transformation_matrix, np.transpose(transcript_positions))[:-1] #this is matrix multiplication
        
        mWmC_cell_center_coordinates= transformed_positions.transpose()  # Transposing the array to 79x2
        print('The transformed center coordinates of all mWmC ROIs in DAPI scale: ', mWmC_cell_center_coordinates)
        print(len(mWmC_cell_center_coordinates)," mWmC cells labled in ",sample_name,region_name)
        plt.scatter(mWmC_cell_center_coordinates[:, 0], mWmC_cell_center_coordinates[:, 1], color = 'r', s =1) 
        plt.imshow(dapi_in_ROI_MIP,cmap='Greys')
        plt.grid(True)  # Adds a grid
        plt.savefig(inputfilepath + "mWmC_on_ROI.png")
        plt.savefig(inputfilepath + "mWmC_on_ROI.pdf")
        plt.show()  # 

        cell_IDs_in_reassigned_masks = []
        for i in range(len(ClusterMap_model.cell_adata)):
           cell_IDs_in_reassigned_masks.append(reassigned_masks[int(ClusterMap_model.cell_adata.obsm['spatial'][i][1])][int(ClusterMap_model.cell_adata.obsm['spatial'][i][0])])       
        print(cell_IDs_in_reassigned_masks)
       
        print('Every cell_IDs_in_reassigned_masks is unique?')
        len(cell_IDs_in_reassigned_masks) == len(list(set(cell_IDs_in_reassigned_masks)))
        
        plt.figure(figsize=(10, 8)) 
        plt.scatter(mWmC_cell_center_coordinates[:,0],mWmC_cell_center_coordinates[:,1], s=3, c='red')

        plt.grid(True)
        plt.gca().invert_yaxis()
        plt.axis('equal')
        plt.show()

        mWmC_positive_cell_IDs_in_reassigned_masks = []
        for i in mWmC_cell_center_coordinates:
            print(reassigned_masks[int(i[1])][int(i[0])])
            if reassigned_masks[int(i[1])][int(i[0])] != 0:
               mWmC_positive_cell_IDs_in_reassigned_masks.append(reassigned_masks[int(i[1])][int(i[0])])

        # Make a mapping from cell_IDs_in_adata to cell_IDs_in_reassigned_masks
        mWmC_positive_cell_IDs_in_adata = []
        for i in mWmC_positive_cell_IDs_in_reassigned_masks:
            mWmC_positive_cell_IDs_in_adata.append(cell_IDs_in_reassigned_masks.index(i))
        print("mWmC positive cell IDs in adata are ",mWmC_positive_cell_IDs_in_adata)

        unique_list = list(set(mWmC_positive_cell_IDs_in_adata))

        str_list = [str(num) for num in unique_list]
        print('There are ',len(str_list), 'mWmC+ cells in this adata')
        print("all", len(str_list),  "mWmC+ cell ID in adata:", str_list)
        
        ClusterMap_model.cell_adata.obs['mWmC'] = ClusterMap_model.cell_adata.obs.index.isin(str_list).astype(str)

        cell_number=mWmC_positive_cell_IDs_in_adata[1]
        plot_cell(inputfilepath=inputfilepath,cell_ID_in_reassigned_masks=cell_number,gene1="Slc17a6",gene2="Gad1")

    adata = ClusterMap_model.cell_adata
    adata.var.index.name = 'genes'
    adata.write_h5ad(inputfilepath+"adata_in_SCROI_"+sample_name+"_"+region_name+".h5ad")
    adata.write_h5ad(adatafilepath+"adata_in_SCROI_"+sample_name+"_"+region_name+".h5ad")

    return(adata)   
