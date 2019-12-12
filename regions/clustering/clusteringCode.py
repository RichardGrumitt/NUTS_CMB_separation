"""Python module to cluster pixels in the sky into regions.
"""
import numpy as np
import matplotlib.pyplot as plt
import healpy as h
from sklearn.cluster import MeanShift, estimate_bandwidth


def compute_naive_betas(map1_name,map2_name,freq1,freq2,offset1=0.,offset2=0.,make_P_from_inputs=False):
    """Compute the simple pixel-by-pixel spectral index.
    Input maps should be in the same coordinate system, have the same Nside and be in the same units.
    
    Parameters
    ----------
    map1_name : str
                Name of first map file
    map2_name : str
                Name of second map file
    freq1 : float
            Frequency of first map [GHz]
    freq2 : float
            Frequency of second map [GHz]
    offset1 : float, 0.
              Zero level offset to add to map1
    offset2 : float, 0.
              Zero level offset to add to map2
              
    Returns
    -------
    beta : array_like
           Healpix map of spectral indices
    """
    if make_P_from_inputs==False:
        map1 = h.read_map(map1_name)
        map2 = h.read_map(map2_name)
    else:
        map1_QU = h.read_map(map1_name,(1,2))
        map2_QU = h.read_map(map2_name,(1,2))
        map1 = np.sqrt(np.power(map1_QU[0],2)+np.power(map1_QU[1],2))
        map2 = np.sqrt(np.power(map2_QU[0],2)+np.power(map2_QU[1],2))
    Nside = h.get_nside(map1)
    Npix = h.nside2npix(Nside)
    mask = np.ones(Npix)
    mask[np.where((map1==h.UNSEEN)|(map2==h.UNSEEN))] = 0.
    map1 += offset1
    map2 += offset2
    beta = np.log(map1/map2)/np.log(freq1/freq2)
    beta[mask==0] = h.UNSEEN
    return beta

def compute_pixel_vectors(Nside):
    """Computes the vectors to the pixel centres for every pixel in a map.
    
    Paramters
    ---------
    Nside : int
            Nside of map
            
    Returns
    -------
    vec : array_like
          Array of vectors to the pixel centres
    """
    Npix = h.nside2npix(Nside)
    Ipix = np.arange(Npix)
    vec = np.asarray(h.pix2vec(Nside, Ipix))
    return vec



def find_meanshift_clusters(beta12,beta23,vec,goodIdx,bandwidth=0.13,min_bin_freq=20):
    """Assign each pixel to a region using the meanshift algorithm.

    Parameters
    ----------
    X : array_like
        Array of values to cluster on
    goodIdx : array_like
              The indices to use
    bandwidth : float, 0.13
                Bandwidth to pass to clustering algorith
    min_bin_freq : int, 20
                   Minimum number of pixels in each region
                   
    Returns
    -------
    labels_map : array_like
                 Healpix map, with each pixel assigned a region.
    """
    NSIDE = h.get_nside(beta12)
    NPIX = h.nside2npix(NSIDE)
    mask = np.ones(NPIX)
    mask[np.where((beta12==h.UNSEEN)|(beta23==h.UNSEEN))] = 0
    X = np.transpose(np.array([beta12,beta23,vec[0],vec[1],vec[2]]))
    X = X[goodIdx]
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, min_bin_freq=min_bin_freq)
    ms.fit(X)
    labels = ms.labels_+1
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("number of estimated clusters : %d" % n_clusters_)
    labels_map = np.ones(NPIX)*h.UNSEEN
    for i in range(0,len(labels)):
        labels_map[goodIdx[0][i]] = labels[i]
    # Unasigned pixels should be added to nearest region
    badIdx = np.where((labels_map==h.UNSEEN)&(mask==1))[0]
    while len(badIdx)>0.:
        lenBadIdx = len(badIdx)
        for idx in badIdx:
            # Find neighbouring pixels and distance to them
            neighbourIdx, neighbourWeights = h.get_interp_weights(NSIDE,vec[0][idx],vec[1][idx])
            # Remove any that are UNSEEN value
            maskedIdx = np.where(labels_map[neighbourIdx]!=h.UNSEEN)[0]
            if len(maskedIdx)==0:
                print('Pixel {0} with no good neighbours'.format(idx))
            elif len(maskedIdx)==1:
                labels_map[idx] = labels_map[neighbourIdx[maskedIdx]]
                print('Pixel {0} with 1 good neighbours'.format(idx))
            else:
                labels_map[idx] = labels_map[neighbourIdx[maskedIdx][np.argmax(neighbourWeights[maskedIdx])]]
        badIdx = np.where((labels_map==h.UNSEEN)&(mask==1))[0]
        if len(badIdx)==lenBadIdx:
            labels_map[badIdx] = 0
    #
    return labels_map


def find_voronoi_clusters(S_map_name,N_map=None,S_map_offset=0,threshold_score = 2.5,min_pixels = 20):
    """Assign each pixel to a region using the Voronoi binning algorithm.
    
    Parameters
    ----------
    S_map_name : str
                 Name of input signal map
    N_map : str, None
            Name of input noise map, if None then noise is set to unity.
    S_map_offset : float, 0
                   Zero level offset to add to signal map
    threshold_score : float, 2.5
                      Target S/N for each region
    min_pixels : int, 20
                 Target minimum number of pixels for each region

    Returns
    -------
    voronoi_map : array_like
                  Healpix map, with each pixel assigned a region.
    distance_map : array_like
                   Healpix map, with the distance from each pixel to its region centroid.
    """
    S_map = h.read_map(S_map_name)
    NPIX = np.size(S_map)
    NSIDE = h.npix2nside(NPIX)
    if N_map==None:
        N_map = np.ones(NPIX)
    #
    mask = np.ones(NPIX)
    mask[S_map==h.UNSEEN] = 0.
    S2N_map = (S_map+S_map_offset)/N_map
    regions_list = []
    centroids_list = []
    score_list = []
    idx_map = np.arange(NPIX)
    bin_map = np.zeros(NPIX)
    for j in range(1,1000):
        print('Creating bin {0}'.format(j))
        # Start new bin
        thisBins_idx = []
        if j==1:
            # If first region, start at maximum signal to noise
            try:
                first_idx = idx_map[np.where((bin_map==0)&(mask==1))][np.argmax(S2N_map[np.where((bin_map==0)&(mask==1))])]
            except ValueError:
                break
        else:
            # If not first region, start at next nearest pixel
            try:
                free_vectors = h.pix2vec(NSIDE,idx_map[np.where((bin_map==0)&(mask==1))])
                pixel_distances = h.rotator.angdist(thisBins_idx_vector,free_vectors)
                first_idx = idx_map[np.where((bin_map==0)&(mask==1))][np.argmin(pixel_distances)]
            except ValueError:
                break
        thisBins_idx.append(first_idx)
        # Find good neighbours
        binComplete=False
        while binComplete==False:
            # List all binned pixel neighbours
            thisBins_neighbours = []
            for i in thisBins_idx:
                thisBins_neighbours.append(h.get_all_neighbours(NSIDE,i))
            thisBins_neighbours = np.unique(np.array(thisBins_neighbours).flatten())
            if np.size(thisBins_neighbours)==0:
                j=-1*j
                break
            # Delete neighbours already in another bin
            thisBins_neighbours = np.delete(thisBins_neighbours,np.where(bin_map[thisBins_neighbours]!=0))
            if np.size(thisBins_neighbours)==0:
                print('Bin: {0} No more neighbours'.format(j))
                j=-1*j
                break
            # Delete neighbours in the mask
            thisBins_neighbours = np.delete(thisBins_neighbours,np.where(mask[thisBins_neighbours]==0))
            if np.size(thisBins_neighbours)==0:
                print('Bin: {0} No more neighbours'.format(j))
                j=-1*j
                break
            # Delete neighbours already in this bin
            for i in thisBins_idx:
                thisBins_neighbours = np.delete(thisBins_neighbours,np.where(thisBins_neighbours==i))
            if np.size(thisBins_neighbours)==0:
                print('Bin: {0} No more neighbours'.format(j))
                j=-1*j
                break
            # Append the bin closest to the centroid
            thisBins_idx_vector = np.average(h.pix2vec(NSIDE,thisBins_idx),axis=1,weights=S2N_map[thisBins_idx])
            thisBins_neighbours_vectors = h.pix2vec(NSIDE,thisBins_neighbours)
            thisBins_angles = h.rotator.angdist(thisBins_idx_vector,thisBins_neighbours_vectors)
            thisBins_idx.append(thisBins_neighbours[np.argmin(thisBins_angles)])
            thisBins_idx_vector = np.average(h.pix2vec(NSIDE,thisBins_idx),axis=1,weights=S2N_map[thisBins_idx])
            # Check the bin's score
            thisBins_score = np.sum(S2N_map[thisBins_idx])
            if (thisBins_score>threshold_score)&(np.size(thisBins_idx)>min_pixels):
                print('Bin: {0}  Score: {1}'.format(j,thisBins_score))
                binComplete=True
        regions_list.append(j)
        centroids_list.append(np.average(h.pix2vec(NSIDE,thisBins_idx),axis=1,weights=S2N_map[thisBins_idx]))
        score_list.append(thisBins_score)
        bin_map[thisBins_idx] = j
    # Now assign each pixel to its closest centroid
    regions_list = np.array(regions_list)
    goodCentroids_vectors = np.array(centroids_list)[np.where(regions_list>0)]
    goodRegions = regions_list[np.where(regions_list>0)]
    voronoi_map = np.zeros(NPIX)
    distance_map = np.zeros(NPIX)
    for i in idx_map:
        # Find angle between this pixel and centroids
        thisPixels_vector = np.array(h.pix2vec(NSIDE,i))
        thisPixels_angles = h.rotator.angdist(thisPixels_vector,np.transpose(goodCentroids_vectors))
        voronoi_map[i] = goodRegions[np.argmin(thisPixels_angles)]
        distance_map[i] = np.min(thisPixels_angles)*180./np.pi
    #
    voronoi_map[mask==0] = h.UNSEEN
    distance_map[mask==0] = h.UNSEEN
    return voronoi_map, distance_map
