import pickle
import numpy as np

#compute attribute graph
NUM_ATTR_REL = 200
def cout_w(prob, num=NUM_ATTR_REL,dim=1):
    prob_weight = prob[:, :num]
    sum_value = np.sum(prob_weight, keepdims=True, axis=dim) + 0.1
    prob_weight = prob_weight / np.repeat(sum_value, prob_weight.shape[dim], axis=dim)
    return prob_weight

def cp_kl(a, b):
    # compute kl diverse
    if np.sum(a) == 0 or np.sum(b) == 0:
        return 1
    sum_ = a * np.log(a / b)
    all_value = [x for x in sum_ if str(x) != 'nan' and str(x) != 'inf']
    kl = np.sum(all_value)
    return kl

def compute_js(attr_prob):
    cls_num = attr_prob.shape[0]
    similarity = np.zeros((cls_num, cls_num))
    similarity[0, 1:] = 1
    similarity[1:, 0] = 1
    for i in range(1, cls_num):
        if i % 50 == 0:
            print('had proccessed {} cls...\n'.format(i))
        for j in range(1, cls_num):
            if i == j:
                similarity[i,j] = 0
            else:
                similarity[i,j] = 0.5 * (cp_kl(attr_prob[i, :], 0.5*(attr_prob[i, :] + attr_prob[j,:]))
                                         + cp_kl(attr_prob[j, :], 0.5*(attr_prob[i, :] + attr_prob[j, :])))
    return similarity


if __name__ == '__main__':
    # We collect coco/voc gt graph by matching classes names with VG.
    # First, you should download attribute/relationship frequency statistic matrix.
    # Second, you should save your dataset classes names. We provide VG names and COCO names.
    data_path = 'Your Path ...'
    vg_class_names_dict = pickle.load(open(data_path + 'vg_3000_name_to_ind.pkl', 'rb'))
    dataset = 'coco'

    vg_ind_cls = {v: k for k, v in vg_class_names_dict.items()}

    vg_class_names = []
    vg_class_names_list = list(vg_class_names_dict)
    for i in vg_class_names_list:
        if i == '__background__':
            vg_class_names.append(i)
        else:
            vg_class_names.append(i.split('.')[0])

    # replace voc name with corresponding to VG
    VOC_class_names = pickle.load(open(data_path + 'VOC_class_names.pkl', 'rb'))
    VOC_class_names = list(VOC_class_names)
    VOC_class_names[VOC_class_names.index('aeroplane')] = 'airplane'
    VOC_class_names[VOC_class_names.index('diningtable')] = 'table'
    VOC_class_names[VOC_class_names.index('motorbike')] = 'motorcycle'
    VOC_class_names[VOC_class_names.index('pottedplant')] = 'plant'
    VOC_class_names[VOC_class_names.index('tvmonitor')] = 'television'

    # replace COCO name with corresponding to VG
    COCO_class_names = pickle.load(open('/data/detection/coco/coco_name.pkl', 'rb'))
    COCO_class_names = list(COCO_class_names)
    COCO_class_names = [n.strip().replace(' ', '_') for n in COCO_class_names]
    COCO_class_names[COCO_class_names.index('tv')] = 'television'
    COCO_class_names[COCO_class_names.index('fire_hydrant')] = 'fireplug'
    COCO_class_names[COCO_class_names.index('stop_sign')] = 'sign'
    COCO_class_names[COCO_class_names.index('handbag')] = 'bag'
    COCO_class_names[COCO_class_names.index('suitcase')] = 'case'
    COCO_class_names[COCO_class_names.index('skis')] = 'ski'
    COCO_class_names[COCO_class_names.index('sports_ball')] = 'ball'
    COCO_class_names[COCO_class_names.index('wine_glass')] = 'goblet'
    COCO_class_names[COCO_class_names.index('orange')] = 'fruit'  # mandarin citrus
    COCO_class_names[COCO_class_names.index('hot_dog')] = 'hotdog'
    COCO_class_names[COCO_class_names.index('donut')] = 'doughnut'
    COCO_class_names[COCO_class_names.index('couch')] = 'sofa'
    COCO_class_names[COCO_class_names.index('remote')] = 'remote_control'
    COCO_class_names[COCO_class_names.index('cell_phone')] = 'mobile'
    COCO_class_names[COCO_class_names.index('dining_table')] = 'table'
    COCO_class_names[COCO_class_names.index('teddy_bear')] = 'teddy'
    COCO_class_names[COCO_class_names.index('hair_drier')] = 'hand_blower'

    index_vg_of_DATASET = []
    if dataset == 'voc':
        for i in COCO_class_names:
            if i == 'potted_plant':
                index_vg_of_DATASET.append(1333)
            else:
                index_vg_of_DATASET.append(vg_class_names.index(i))
    elif dataset == 'coco':
        for i in COCO_class_names:
            index_vg_of_DATASET.append(vg_class_names.index(i))

    # Attribute graph
    graph_a = pickle.load(open(data_path + 'vg_attr_frequency_3000.pkl', 'rb'))
    new_graph = graph_a[index_vg_of_DATASET,]
    new_graph = cout_w(new_graph, num=len(new_graph))
    new_graph = compute_js(new_graph)
    new_graph = 1 - new_graph
    pickle.dump(new_graph, open(data_path + dataset + '_graph_a.pkl', 'wb'))

    # Relation graph
    graph_r = pickle.load(open(data_path + 'vg_pair_frequency_3000.pkl', 'rb'))
    index_vg_of_DATASET_ = [x for x in index_vg_of_DATASET[1:]]
    # graph_r_COCO = graph_r[[x for x in index_vg_of_COCO_big],]
    new_graph = graph_r[index_vg_of_DATASET_, :]
    new_graph = new_graph[:, index_vg_of_DATASET_]

    relation_matrix = new_graph + new_graph.transpose()
    relation_matrix_row_sum = relation_matrix.sum(1)
    num_classes = len(index_vg_of_DATASET) - 1
    for i in range(num_classes):
        relation_matrix[i, i] = relation_matrix_row_sum[i] + 1
    prob_relation_matrix = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            prob_relation_matrix[i, j] = relation_matrix[i, j] / (
                    np.sqrt(relation_matrix[i, i]) * np.sqrt(relation_matrix[j, j]))
    graph_r_ = np.zeros((num_classes + 1, num_classes + 1))
    graph_r_[1:, 1:] = prob_relation_matrix
    pickle.dump(graph_r_, open(data_path + dataset + '_graph_r.pkl', 'wb'))

