之前的 run_mapanything.py 没实现--apply_mask False 就生成了一版 mapanything_predictions_playroom_full.pt。需要重新 run_mapanything.py 生成并可视化检验效果。

mask 作用的原理 （将哪些作为mask

AI说“不是你写的代码有 bug，是 MapAnything 对这个场景 mask 全判死 + 它把失效点清零了。” 但是之前truck的数据集没有把mask“全判死” 但也是生成出了球壳形的点云。

进一步检查 run_mapanything.py 、可视化相关代码 是否存在逻辑问题，尤其关注类似坐标系转换这种问题，毕竟可视化的点云结果呈现球壳状也是比较特别的形状。