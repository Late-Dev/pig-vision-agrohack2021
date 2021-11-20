import h5py as h5
import os.path as fs


"""
Для минутного ролика имеем один файл. Внутри файла группы с 
Frame_i, где i -- это номер фрейма по порядку. 

num_pigs -- число свиней на кадре. list <Int [1]>
boxs -- список из боксов на каждом фрейме; list <array [num_pigs, 4] >, [x_min, y_min, x_max, y_max]
masks -- список из списка вырезанных масок (избавляемся от лишних нулей, 
    оставляем только то, что внутри соответствуюещй коробки) list < list [num_pigs, x_shape, y_shape] >

Для трекинга, boxs и masks сразу сохраняем в соответвтующем порядке.

"""
def savePredict(Path, Name, boxs, masks, num_pigs, track_ids):
    ff = h5.File(fs.join(Path, Name), 'w')
    num_frames = len(boxs) # количество фреймов
    for frame_iter in range(num_frames):
        grp = ff.create_group("Frame_%d"%frame_iter)
        grp.create_dataset('boxs', data = boxs[frame_iter])
        grp.create_dataset('num_pigs', data = num_pigs[frame_iter])
        grp.create_dataset('track_ids', data = track_ids[frame_iter])
        subgrp = grp.create_group("PolyMasks")
        for pig_inter in range(num_pigs[frame_iter]):
            subgrp.create_dataset('polymask_%d'%pig_inter, data = masks[frame_iter][pig_inter])
    ff.close()
