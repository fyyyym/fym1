# fym1

# Weekly Plan(12.24)

- 训练出带未来mos-mask的网络。
- 整合运动残差进入网络。

:blush:

训练了50epoche atppnet   /home/ugvc4090/ATPPNet-ori/atppnet/runs/ATPPNet_20241219_175636
| 列1    | 列2    | 列3   |
|——|——|—–|
| 内容1  | 内容2  | 内容3 |
| 内容4  | 内容5  | 内容6 |


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

再训练10
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃         Test metric         ┃        DataLoader 0         ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│   test/chamfer_distance_0   │     0.22422951459884644     │
│   test/chamfer_distance_1   │     0.27779147028923035     │
│   test/chamfer_distance_2   │      0.345889151096344      │
│   test/chamfer_distance_3   │     0.41676273941993713     │
│   test/chamfer_distance_4   │     0.4995008707046509      │
│ test/final_chamfer_distance │     0.4995008707046509      │
│     test/inference_time     │     0.03369585797190666     │
│       test/l1_loss_0        │     0.48343712091445923     │
│       test/l1_loss_1        │     0.5939206480979919      │
│       test/l1_loss_2        │     0.7016581296920776      │
│       test/l1_loss_3        │     0.8022783994674683      │
│       test/l1_loss_4        │     0.8979970216751099      │
│       test/loss_mask        │     0.3090978264808655      │
│    test/loss_range_view     │      0.695857584476471      │
│ test/mean_chamfer_distance  │     0.35283374786376953     │
└─────────────────────────────┴─────────────────────────────┘
test
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃         Test metric         ┃        DataLoader 0         ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│   test/chamfer_distance_0   │     0.22600868344306946     │
│   test/chamfer_distance_1   │     0.27150392532348633     │
│   test/chamfer_distance_2   │     0.3281521201133728      │
│   test/chamfer_distance_3   │     0.3901638686656952      │
│   test/chamfer_distance_4   │     0.4589468538761139      │
│ test/final_chamfer_distance │     0.4589468538761139      │
│     test/inference_time     │     0.2266755998134613      │
│       test/l1_loss_0        │     0.4686991572380066      │
│       test/l1_loss_1        │     0.5706849098205566      │
│       test/l1_loss_2        │     0.6675057411193848      │
│       test/l1_loss_3        │     0.7599059343338013      │
│       test/l1_loss_4        │     0.8510318398475647      │
│       test/loss_mask        │     0.3045000731945038      │
│    test/loss_range_view     │     0.6635656952857971      │
│ test/mean_chamfer_distance  │     0.33495479822158813     │
└─────────────────────────────┴─────────────────────────────┘

学习怎么用运动信息







┃         Test metric         ┃        DataLoader 0         ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│   test/chamfer_distance_0   │     0.4369220435619354      │
│   test/chamfer_distance_1   │     0.6083724498748779      │
│   test/chamfer_distance_2   │     0.7891181707382202      │
│   test/chamfer_distance_3   │      0.994104266166687      │
│   test/chamfer_distance_4   │     1.2346855401992798      │
│ test/final_chamfer_distance │     1.2346855401992798      │
│     test/inference_time     │     0.03643617779016495     │
│       test/l1_loss_0        │      0.473813533782959      │
│       test/l1_loss_1        │     0.5867612957954407      │
│       test/l1_loss_2        │     0.7012622356414795      │
│       test/l1_loss_3        │      0.80788654088974       │
│       test/l1_loss_4        │     0.9087781310081482      │
│       test/loss_mask        │     0.3085906207561493      │
│    test/loss_range_view     │     0.6957002282142639      │
│ test/mean_chamfer_distance  │     0.8126397132873535      │



