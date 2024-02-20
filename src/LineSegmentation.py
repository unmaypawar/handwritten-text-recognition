from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage.filters import sobel
import numpy as np
from heapq import *
from skimage.filters import threshold_otsu
from skimage.util import invert
from PIL import Image
import os


def horizontal_projections(sobel_image):
    return np.sum(sobel_image, axis=1)

def find_peak_regions(hpp, divider=2):
    threshold = (np.max(hpp)-np.min(hpp))/divider
    peaks = []
    peaks_index = []
    for i, hppv in enumerate(hpp):
        if hppv < threshold:
            peaks.append([i, hppv])
    return peaks

def get_hpp_walking_regions(peaks_index):
    hpp_clusters = []
    cluster = []
    for index, value in enumerate(peaks_index):
        cluster.append(value)

        if index < len(peaks_index)-1 and peaks_index[index+1] - value > 1:
            hpp_clusters.append(cluster)
            cluster = []

        #get the last cluster
        if index == len(peaks_index)-1:
            hpp_clusters.append(cluster)
            cluster = []
            
    return hpp_clusters

def heuristic(a, b):
    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2

def astar(array, start, goal):

    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []

    heappush(oheap, (fscore[start], start))
    
    while oheap:

        current = heappop(oheap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j            
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:                
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue
                
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
                
            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heappush(oheap, (fscore[neighbor], neighbor))
                
    return []

def get_binary(img):
    mean = np.mean(img)
    if mean == 0.0 or mean == 1.0:
        return img

    thresh = threshold_otsu(img)
    binary = img <= thresh
    binary = binary*1
    return binary

def path_exists(window_image):
    #very basic check first then proceed to A* check
    if 0 in horizontal_projections(window_image):
        return True
    
    padded_window = np.zeros((window_image.shape[0],1))
    world_map = np.hstack((padded_window, np.hstack((window_image,padded_window)) ) )
    path = np.array(astar(world_map, (int(world_map.shape[0]/2), 0), (int(world_map.shape[0]/2), world_map.shape[1])))
    if len(path) > 0:
        return True
    
    return False

def get_road_block_regions(nmap):
    road_blocks = []
    needtobreak = False
    
    for col in range(nmap.shape[1]):
        start = col
        end = col+20
        if end > nmap.shape[1]-1:
            end = nmap.shape[1]-1
            needtobreak = True

        if path_exists(nmap[:, start:end]) == False:
            road_blocks.append(col)

        if needtobreak == True:
            break
            
    return road_blocks

def group_the_road_blocks(road_blocks):
    #group the road blocks
    road_blocks_cluster_groups = []
    road_blocks_cluster = []
    size = len(road_blocks)
    for index, value in enumerate(road_blocks):
        road_blocks_cluster.append(value)
        if index < size-1 and (road_blocks[index+1] - road_blocks[index]) > 1:
            road_blocks_cluster_groups.append([road_blocks_cluster[0], road_blocks_cluster[len(road_blocks_cluster)-1]])
            road_blocks_cluster = []

        if index == size-1 and len(road_blocks_cluster) > 0:
            road_blocks_cluster_groups.append([road_blocks_cluster[0], road_blocks_cluster[len(road_blocks_cluster)-1]])
            road_blocks_cluster = []

    return road_blocks_cluster_groups

def extract_line_from_image(image, lower_line, upper_line):
    lower_boundary = np.min(lower_line[:, 0])
    upper_boundary = np.min(upper_line[:, 0])
    img_copy = np.copy(image)
    r, c = img_copy.shape
    for index in range(c-1):
        img_copy[0:lower_line[index, 0], index] = 255
        img_copy[upper_line[index, 0]:r, index] = 255
    
    return img_copy[lower_boundary:upper_boundary, :]

def line_segmentation(img):
    APP_ROOT = os.path.dirname(os.path.abspath(__file__))
    img = imread(img)
    sobel_image = sobel(img)
    hpp = horizontal_projections(sobel_image)
    peaks = find_peak_regions(hpp)

    peaks_index = np.array(peaks)[:,0].astype(int)

    segmented_img = np.copy(img)
    r,c = segmented_img.shape
    for ri in range(r):
        if ri in peaks_index:
            segmented_img[ri, :] = 0

    hpp_clusters = get_hpp_walking_regions(peaks_index)
    binary_image = get_binary(img)
    for cluster_of_interest in hpp_clusters:
        nmap = binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest)-1],:]
        road_blocks = get_road_block_regions(nmap)
        road_blocks_cluster_groups = group_the_road_blocks(road_blocks)
        #create the doorways
        for index, road_blocks in enumerate(road_blocks_cluster_groups):
            window_image = nmap[:, road_blocks[0]: road_blocks[1]+10]
            binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest)-1],:][:, road_blocks[0]: road_blocks[1]+10][int(window_image.shape[0]/2),:] *= 0
    line_segments = []
    for i, cluster_of_interest in enumerate(hpp_clusters):
        nmap = binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest)-1],:]
        path = np.array(astar(nmap, (int(nmap.shape[0]/2), 0), (int(nmap.shape[0]/2),nmap.shape[1]-1)))
        offset_from_top = cluster_of_interest[0]
        path[:,0] += offset_from_top
        line_segments.append(path)
    cluster_of_interest = hpp_clusters[1]
    offset_from_top = cluster_of_interest[0]
    nmap = binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest)-1],:]
    plt.figure(figsize=(20,20))
    plt.imshow(invert(nmap), cmap="gray")
    
    path = np.array(astar(nmap, (int(nmap.shape[0]/2), 0), (int(nmap.shape[0]/2),nmap.shape[1]-1)))
    plt.plot(path[:,1], path[:,0])
    
    offset_from_top = cluster_of_interest[0]
    fig, ax = plt.subplots(figsize=(20,10), ncols=2)
    for path in line_segments:
        ax[1].plot((path[:,1]), path[:,0])
    ax[1].axis("off")
    ax[0].axis("off")
    ax[1].imshow(img, cmap="gray")
    ax[0].imshow(img, cmap="gray")
    last_bottom_row = np.flip(np.column_stack(((np.ones((img.shape[1],))*img.shape[0]), np.arange(img.shape[1]))).astype(int), axis=0)
    line_segments.append(last_bottom_row)
    line_images = []
    line_count = len(line_segments)
    fig, ax = plt.subplots(figsize=(10,10), nrows=line_count-1)
    for line_index in range(line_count-1):
        line_image = extract_line_from_image(img, line_segments[line_index], line_segments[line_index+1])
        line_images.append(line_image)
        ax[line_index].imshow(line_image, cmap="gray")
    target = os.path.join(APP_ROOT, 'static/')
    for i in range(line_count-1):
        image_save = Image.fromarray(line_images[i])
        savefname = "segment (" + str(i+1) + ").jpg"
        destination = "/".join([target, savefname])
        image_save.save(destination)
    return line_count
