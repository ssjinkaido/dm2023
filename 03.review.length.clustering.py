clusters = []
max_length = 0
for sent in data_processed:
    clusters.append([(sent, len(sent.split()))])
    
def calc_dist_between_cluster(list_clusters_i, list_clusters_j):
    min_dist = 9999
    for i in list_clusters_i:
        for j in list_clusters_j:
            min_dist = min(abs(i[1] - j[1]), min_dist)
    return min_dist
    
while(len(clusters)>3):
    min_dist = 9999
    index_i = 0
    index_j = 0
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            min_dist_between_cluster = calc_dist_between_cluster(clusters[i], clusters[j])
            if min_dist_between_cluster < min_dist:
                index_i = i
                index_j = j
                min_dist = min_dist_between_cluster
    for i in range(len(clusters[index_j])):
        clusters[index_i].append(clusters[index_j][i])
    del clusters[index_j]

for i in range(len(clusters)):
    print(f"Length cluster {i}: {len(clusters[i])}")
