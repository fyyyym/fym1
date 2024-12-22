# fym1

# Weekly Plan(12.19)
1.完成motio



训练了50epoche atppnet   /home/ugvc4090/ATPPNet-ori/atppnet/runs/ATPPNet_20241219_175636

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃         Test metric         ┃        DataLoader 0         ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│   test/chamfer_distance_0   │     0.3734810948371887      │
│   test/chamfer_distance_1   │     0.5312111973762512      │
│   test/chamfer_distance_2   │      0.711632251739502      │
│   test/chamfer_distance_3   │     0.9161503911018372      │
│   test/chamfer_distance_4   │     1.1375048160552979      │
│ test/final_chamfer_distance │     1.1375048160552979      │
│     test/inference_time     │     0.03423868864774704     │
│       test/l1_loss_0        │     0.45316389203071594     │
│       test/l1_loss_1        │     0.5593820810317993      │
│       test/l1_loss_2        │     0.6678876280784607      │
│       test/l1_loss_3        │     0.7678834199905396      │
│       test/l1_loss_4        │     0.8639376759529114      │
│       test/loss_mask        │     0.3052016794681549      │
│    test/loss_range_view     │     0.6624512076377869      │
│ test/mean_chamfer_distance  │     0.7339957356452942      │
└─────────────────────────────┴─────────────────────────────┘
又训练了10轮带chamferdistance的 没有达到论文中0.335的要求

│   test/chamfer_distance_0   │     0.2355964034795761      │
│   test/chamfer_distance_1   │     0.28305986523628235     │
│   test/chamfer_distance_2   │     0.3481317460536957      │
│   test/chamfer_distance_3   │     0.41803231835365295     │
│   test/chamfer_distance_4   │     0.4933539927005768      │
│ test/final_chamfer_distance │     0.4933539927005768      │
│     test/inference_time     │     0.04013470560312271     │
│       test/l1_loss_0        │     0.4967802166938782      │
│       test/l1_loss_1        │     0.6056380271911621      │
│       test/l1_loss_2        │     0.7152851819992065      │
│       test/l1_loss_3        │     0.8206812143325806      │
│       test/l1_loss_4        │     0.9236515760421753      │
│       test/loss_mask        │     0.3086422383785248      │
│    test/loss_range_view     │     0.7124070525169373      │
│ test/mean_chamfer_distance  │     0.3556349575519562  

学习怎么用运动信息
