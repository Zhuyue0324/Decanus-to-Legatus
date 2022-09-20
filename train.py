import os
from PIL import Image
import torch.optim as optim
from options import Options
from utils.utils import *
import math

from model import *

# options
# print("Configurating parameters...")
options = Options()
opts = options.parse()


class Trainer:
    def __init__(self, options, pretrain=False):
        print(' ')
        print('-----------------------------------------------------------')
        print('Initializing')
        print('-----------------------------------------------------------')
        print(' ')
        self.opt = options
        self.distribution_seed = 'handcrafted'
        self.distribution_0_path = './distribution/' + self.distribution_seed + '/distribution_train_2d.txt'
        # print('The initial distribution is from', self.distribution_seed)
        self.task = self.opt.task
        self.datasets_binary = self.opt.dataset  # ['h36m', 'coco', 'mpii', 'mpi_inf_3dhp'], 4 = [0, 1, 0, 0]
        self.use_dataset = []
        for i in range(4):
            self.use_dataset.append(int((self.datasets_binary // (8 / (2 ** i))) % 2))
        self.dataset_names = ['h36m', 'coco', 'mpii', 'mpi_inf_3dhp']
        self.dataset = []
        for i in range(len(self.use_dataset)):
            if self.use_dataset[i] > 0:
                self.dataset.append(self.dataset_names[i])
        self.data_path = self.opt.data_path
        self.weights_path = self.opt.weights_path
        self.subfolder_path = self.opt.subfolder_path
        self.lr_rate_lifter = self.opt.pose_lr_rate
        self.alpha_distribution_diffution = self.opt.diffusion_speed
        self.initial_diffusion_steps = self.opt.prediffusion
        self.batch_size = self.opt.batch_size
        self.train_epoch = self.opt.train_epoch
        self.num_key_points = self.opt.num_key_points
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_elementwiseloss = torch.zeros(self.num_key_points-1).to(self.device)
        self.print_frequence = 500
        self.use_indicator = True
        self.num_view = 4
        self.coefficients = [1, 0.1, 0, 0]  # LX2d, L3d, Lx3d, Lcc (Lcc<0 == Samplewise, Lcc>=0 == Batchwise)

        if self.use_indicator:
            self.net = Lifter(self.num_key_points, terms=3)
        else:
            self.net = Lifter(self.num_key_points, terms=2)
        self.net.to(self.device)
        if pretrain:
            if os.path.exists(self.weights_path + self.subfolder_path+'lifter.pth'):
                self.net.load_state_dict(torch.load(self.weights_path+self.subfolder_path+'lifter.pth',
                                                    map_location=self.device))
                # print("Loaded weights at:" + self.weights_path + self.subfolder_path+'lifter.pth')
                # else:
                # print("No pretrained weights found at:" + self.weights_path + self.subfolder_path+'lifter.pth')

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr_rate_lifter, weight_decay=1e-5)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20], gamma=0.1)
        self.net.train()

        self.current_dtb = torch.ones(50, 50, 17, 3).to(self.device)
        self.target_dtb = torch.zeros(50, 50, 17, 3).to(self.device)
        self.target_value = torch.zeros(51, 2, 17, 3).to(self.device)
        self.root_current = torch.ones(50, 3).to(self.device)
        self.root_target = torch.zeros(50, 3).to(self.device)
        self.root_values = torch.zeros(51, 3).to(self.device)

        os.makedirs('./distribution/')
        os.makedirs('./examples/')

    def compute_crafted_distribution(self, full_range=True):
        print(' ')
        print('-----------------------------------------------------------')
        print('Reproducing distribution record')
        print('-----------------------------------------------------------')
        print(' ')
        save_path = './distribution/handcrafted/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Since we can't provide the whole list of human3.6m, we hardcode the examples here instead of choosing by
        # another algorithm in practice.
        if True:
            example_2D = torch.tensor([[[662.4335, 371.8898],
                                        [689.6948, 379.4560],
                                        [696.5954, 452.9332],
                                        [706.1444, 532.5082],
                                        [635.4212, 364.4296],
                                        [653.7952, 452.6448],
                                        [667.6879, 538.7413],
                                        [677.1640, 341.0952],
                                        [681.5732, 314.2727],
                                        [684.1469, 307.9488],
                                        [681.8423, 287.5162],
                                        [655.3397, 323.0555],
                                        [647.0711, 373.0593],
                                        [683.0306, 377.3251],
                                        [707.5604, 323.4281],
                                        [698.7002, 369.6584],
                                        [684.0424, 324.8250]],

                                       [[485.2759, 338.9261],
                                        [452.9919, 336.2268],
                                        [444.2509, 440.6290],
                                        [432.5321, 531.7878],
                                        [516.7273, 341.5996],
                                        [508.7292, 444.8681],
                                        [496.7346, 535.5894],
                                        [490.6497, 276.2723],
                                        [498.4362, 215.5693],
                                        [506.5020, 191.9257],
                                        [502.0471, 163.7975],
                                        [532.3583, 218.2748],
                                        [586.2863, 238.3002],
                                        [639.4234, 244.5292],
                                        [457.0483, 215.9915],
                                        [387.3246, 230.5897],
                                        [328.0116, 233.0387]],

                                       [[772.1569, 374.3240],
                                        [798.8859, 371.5023],
                                        [806.9502, 464.4076],
                                        [795.7050, 544.4542],
                                        [745.8590, 377.1141],
                                        [761.0505, 467.7678],
                                        [753.8381, 565.1971],
                                        [761.3439, 326.0638],
                                        [750.9464, 268.5081],
                                        [755.5501, 264.2765],
                                        [770.6581, 242.5318],
                                        [720.1698, 282.2568],
                                        [704.9880, 334.3701],
                                        [715.1940, 324.5064],
                                        [781.0056, 264.5300],
                                        [788.7883, 309.7020],
                                        [750.2330, 316.8806]],

                                       [[395.4859, 430.3580],
                                        [378.3794, 437.7563],
                                        [371.7729, 594.3510],
                                        [324.5486, 712.9761],
                                        [411.2305, 423.5585],
                                        [467.0071, 573.5292],
                                        [406.4569, 704.3390],
                                        [388.3350, 347.4747],
                                        [399.9658, 250.9077],
                                        [430.7520, 223.1804],
                                        [401.6889, 189.3328],
                                        [399.4217, 265.3285],
                                        [387.7441, 349.2411],
                                        [388.6842, 420.2994],
                                        [386.1709, 279.4013],
                                        [383.5342, 388.5079],
                                        [420.9963, 479.8285]],

                                       [[563.7396, 391.7348],
                                        [541.5422, 389.7047],
                                        [541.8900, 476.0495],
                                        [544.4676, 557.5845],
                                        [586.4718, 393.8293],
                                        [580.2825, 480.6508],
                                        [580.5449, 564.3677],
                                        [546.4189, 349.5203],
                                        [509.4317, 314.6069],
                                        [492.8282, 306.4030],
                                        [492.0288, 283.7244],
                                        [536.7759, 307.7684],
                                        [582.3358, 340.6722],
                                        [592.2892, 389.4486],
                                        [501.4910, 337.4224],
                                        [502.0986, 391.8727],
                                        [499.3739, 440.6059]],

                                       [[780.4014, 389.8845],
                                        [811.3442, 390.8817],
                                        [798.4583, 492.3292],
                                        [801.6699, 585.3043],
                                        [749.4454, 388.9080],
                                        [750.6831, 492.4975],
                                        [755.0242, 600.5839],
                                        [788.3258, 332.8404],
                                        [791.6467, 266.3932],
                                        [785.1531, 253.2276],
                                        [795.4241, 224.9102],
                                        [758.5851, 285.5929],
                                        [733.5474, 351.3391],
                                        [714.9796, 404.0079],
                                        [825.0784, 281.4691],
                                        [831.3742, 314.2090],
                                        [809.1871, 266.5663]],

                                       [[815.2280, 366.2481],
                                        [842.6507, 372.7012],
                                        [849.9867, 456.4425],
                                        [838.1296, 514.1774],
                                        [788.2539, 359.9295],
                                        [805.6887, 450.3355],
                                        [815.6716, 539.7364],
                                        [823.5925, 321.7570],
                                        [824.3080, 266.8686],
                                        [814.8609, 250.5880],
                                        [823.0101, 232.4748],
                                        [796.0311, 278.0941],
                                        [768.8304, 331.0913],
                                        [744.5699, 378.2190],
                                        [851.5624, 280.1576],
                                        [867.2034, 337.5356],
                                        [877.3414, 386.2402]],

                                       [[444.2866, 381.5990],
                                        [436.9461, 383.2907],
                                        [485.4850, 496.4909],
                                        [507.5052, 605.4858],
                                        [451.1955, 380.0089],
                                        [464.7396, 497.2232],
                                        [446.2258, 607.8682],
                                        [459.5965, 319.0294],
                                        [516.4972, 290.2053],
                                        [542.4232, 296.4131],
                                        [552.1044, 267.8193],
                                        [508.4008, 295.1160],
                                        [459.6489, 323.1567],
                                        [424.2676, 366.5683],
                                        [500.2243, 301.6728],
                                        [480.2530, 349.3615],
                                        [496.0662, 387.2318]],

                                       [[445.1660, 435.4021],
                                        [472.5324, 432.5000],
                                        [466.6479, 526.9640],
                                        [467.9853, 623.0856],
                                        [417.3782, 438.3677],
                                        [422.0746, 533.8879],
                                        [426.1478, 631.2753],
                                        [447.2466, 386.4193],
                                        [444.9064, 329.6348],
                                        [444.3995, 311.5171],
                                        [446.9788, 289.3486],
                                        [414.4114, 343.3675],
                                        [395.7997, 393.3565],
                                        [418.8385, 343.8594],
                                        [476.3038, 341.2231],
                                        [499.1226, 398.2242],
                                        [509.9825, 448.3003]],

                                       [[402.1885, 478.2166],
                                        [428.3311, 478.5809],
                                        [401.9265, 496.8062],
                                        [394.3688, 578.3749],
                                        [375.6194, 477.8522],
                                        [353.4426, 498.9314],
                                        [360.7734, 581.2982],
                                        [410.0322, 436.1097],
                                        [410.4901, 383.3456],
                                        [409.6763, 375.5935],
                                        [405.2279, 354.4612],
                                        [384.3320, 394.4043],
                                        [357.4247, 442.5352],
                                        [397.6335, 442.2373],
                                        [435.9616, 395.2457],
                                        [423.7911, 435.8816],
                                        [404.3315, 391.6076]]])

        # Manual annotate front/back, also hardcode here
        if True:
            annot = []
            annot.append(torch.tensor([[0, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1]]))
            annot.append(torch.tensor([[0, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1]]))
            annot.append(torch.tensor([[0, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, -1, 1, 1]]))
            annot.append(torch.tensor([[0, -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1]]))
            annot.append(torch.tensor([[0, 1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 1, 1, 1]]))
            annot.append(torch.tensor([[0, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1]]))
            annot.append(torch.tensor([[0, -1, 1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, 1, 1]]))
            annot.append(torch.tensor([[0, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1]]))
            annot.append(torch.tensor([[0, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1]]))
            annot.append(torch.tensor([[0, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1]]))
            annot = torch.cat(annot, dim=0).unsqueeze(2)

        records = craft_lifted_3d_poses(example_2D, depth_sign=annot)
        boundary = craft_generation_boundary()

        records = records - records[:, 0:1, :]
        records /= forbenius_norm_multi_dim(records)
        records = torch.cat((records[..., 0:1], records[..., 2:3], -records[..., 1:2]), dim=2) * 0.2
        records = find_17_rho_theta_phi(records)

        rmins = boundary[..., 0]
        rmaxs = boundary[..., 1]

        for i in range(records.shape[0]):
            dd = rho_theta_phi_to_skeleton(records[i:(i+1), ...])
            dd = torch.cat((dd[:, :, 0:1], -dd[:, :, 2:3], dd[:, :, 1:2]), 2)
            draw_skeleton(dd, plt_show=False, save_path=save_path+'sample_pose_'+str(i)+'.jpg')
            draw_skeleton(dd[..., :2], plt_show=False, save_path=save_path+'sample_pose_'+str(i)+'_2D.jpg')

        histname = ['rho', 'theta', 'phi']
        list_joint_pairs = [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (8, 10), (8, 11),
                            (4, 11), (11, 12), (12, 13), (11, 14), (14, 15), (15, 16)]
        hist_record = np.zeros((50*50+51*2, 3*len(list_joint_pairs)+1))

        for i in range(len(list_joint_pairs)):
            m, n = list_joint_pairs[i]
            for j in range(3):
                fig = plt.figure()
                if full_range:
                    plt.hist2d(records[:, m, j].numpy(), records[:, n, j].numpy(), bins=50,
                               range=[[rmins[m, j].item(), rmaxs[m, j].item()],
                                      [rmins[n, j].item(), rmaxs[n, j].item()]],
                               cmap=plt.cm.jet)
                    for k in range(50):
                        hist_record[(k*50):(k*50+50), 3*i+j] = \
                            plt.hist2d(records[:, m, j].numpy(), records[:, n, j].numpy(), bins=50,
                                       range=[[rmins[m, j].item(), rmaxs[m, j].item()],
                                              [rmins[n, j].item(), rmaxs[n, j].item()]],
                                       cmap=plt.cm.jet)[0][k, :]
                    hist_record[2500:2551, 3*i+j] = plt.hist2d(records[:, m, j].numpy(), records[:, n, j].numpy(),
                                                               bins=50, range=[[rmins[m, j].item(), rmaxs[m, j].item()],
                                                                               [rmins[n, j].item(), rmaxs[n, j].item()]],
                                                               cmap=plt.cm.jet)[1]
                    hist_record[2551:2602, 3*i+j] = plt.hist2d(records[:, m, j].numpy(), records[:, n, j].numpy(),
                                                               bins=50, range=[[rmins[m, j].item(), rmaxs[m, j].item()],
                                                                               [rmins[n, j].item(), rmaxs[n, j].item()]],
                                                               cmap=plt.cm.jet)[2]
                else:
                    plt.hist2d(records[:, m, j].numpy(), records[:, n, j].numpy(), bins=50, cmap=plt.cm.jet)
                    for k in range(50):
                        hist_record[(k*50):(k*50+50), 3*i+j] = plt.hist2d(records[:, m, j].numpy(),
                                                                          records[:, n, j].numpy(), bins=50,
                                                                          cmap=plt.cm.jet)[0][k, :]
                    hist_record[2500:2551, 3*i+j] = plt.hist2d(records[:, m, j].numpy(), records[:, n, j].numpy(),
                                                               bins=50, cmap=plt.cm.jet)[1]
                    hist_record[2551:2602, 3*i+j] = plt.hist2d(records[:, m, j].numpy(), records[:, n, j].numpy(),
                                                               bins=50, cmap=plt.cm.jet)[2]
                plt.xlabel('histogram_' + str(m)+'_' + str(n)+'_' + histname[j])
                fig.savefig(save_path+'histogram_' + str(m)+'_' + str(n)+'_' + histname[j]+'.jpg')
        for j in range(3):
            fig = plt.figure()
            if full_range:
                plt.hist(records[:, 0, j].numpy(), bins=50, range=(rmins[0, j].item(), rmaxs[0, j].item()),
                         color='Green', alpha=0.5)
                hist_record[(j*101):(j*101+50), 3*len(list_joint_pairs)] = \
                    plt.hist(records[:, 0, j].numpy(), bins=50, range=(rmins[0, j].item(), rmaxs[0, j].item()),
                             color='Green', alpha=0.5)[0]
                hist_record[(j*101+50):(j*101+101), 3*len(list_joint_pairs)] = \
                    plt.hist(records[:, 0, j].numpy(), bins=50, range=(rmins[0, j].item(), rmaxs[0, j].item()),
                             color='Green', alpha=0.5)[1]
            else:
                plt.hist(records[:, 0, j].numpy(), bins=50, color='Green', alpha=0.5)
                hist_record[(j*101):(j*101+50), 3*len(list_joint_pairs)] = \
                    plt.hist(records[:, 0, j].numpy(), bins=50, color='Green', alpha=0.5)[0]
                hist_record[(j*101+50):(j*101+101), 3*len(list_joint_pairs)] = \
                    plt.hist(records[:, 0, j].numpy(), bins=50, color='Green', alpha=0.5)[1]
            plt.xlabel('histogram_0_' + histname[j])
            fig.savefig(save_path+'histogram_0_' + histname[j] + '.jpg')

        np.savetxt(save_path+'distribution_train_2d.txt', hist_record, delimiter=';')

        print('The distribution is saved inside ./distribution/handcrafted')

        return 0

    def init_distribution(self):
        distribution = np.loadtxt(self.distribution_0_path, delimiter=';')
        distribution = torch.from_numpy(distribution).to(self.device)

        self.current_dtb = torch.ones(50, 50, 17, 3).to(self.device)
        self.target_dtb = torch.zeros(50, 50, 17, 3).to(self.device)
        self.target_value = torch.zeros(51, 2, 17, 3).to(self.device)
        self.root_current = torch.ones(50, 3).to(self.device)
        self.root_target = torch.zeros(50, 3).to(self.device)
        self.root_values = torch.zeros(51, 3).to(self.device)

        for i in range(17):
            for j in range(3):
                for k in range(50):
                    self.target_dtb[k, :, i, j] = distribution[(k * 50):(k * 50 + 50), 3 * i + j]
                self.target_value[:, 0, i, j] = distribution[2500:2551, 3 * i + j]
                self.target_value[:, 1, i, j] = distribution[2551:2602, 3 * i + j]
        for j in range(3):
            self.root_target[:, j] = distribution[(j * 101):(j * 101 + 50), 3 * 17]
            self.root_values[:, j] = distribution[(j * 101 + 50):(j * 101 + 101), 3 * 17]

    def show_generation_sample(self, size=10):
        print(' ')
        print('-----------------------------------------------------------')
        print('Drawing generation samples')
        print('-----------------------------------------------------------')
        print(' ')
        trainer.init_distribution()
        pose3d = create_skeleton_17(trainer.current_dtb, trainer.target_dtb, trainer.target_value,
                                    trainer.root_current, trainer.root_target, trainer.root_values, batchsize=size)
        for i in range(size):
            draw_skeleton(pose3d[i:(i + 1), ...], plt_show=False, save_path='./examples/sample-pose-' + str(i) + '.png')
        print('The generation samples are saved inside ./examples')

    def epoch_train_synthetic(self, synt_extrinsic, records=None):
        loss = torch.zeros(self.num_key_points-1).to(self.device)

        pose3d = create_skeleton_17(self.current_dtb, self.target_dtb, self.target_value,
                                    self.root_current, self.root_target, self.root_values, batchsize=self.batch_size)
        pose3d = (pose3d-pose3d[:, 0:1, :])[:, 1:, :]
        pose3d = pose3d.repeat(self.num_view, 1, 1)
        synt_rotations = rot_from_axisangle(synt_extrinsic)
        pose3d = torch.transpose(torch.matmul(synt_rotations, torch.transpose(pose3d, 1, 2)), 1, 2)
        gt_X1 = pose3d/forbenius_norm_multi_dim(pose3d)
        pose2d = pose3d[:, :, :2].reshape(self.num_view*self.batch_size, -1)
        gt_2d = pose2d / forbenius_norm_multi_dim(pose2d)

        if self.use_indicator:
            input_2d = torch.cat((gt_2d.reshape(self.num_view*self.batch_size, self.num_key_points - 1, 2),
                                  (torch.ones(self.num_view*self.batch_size, self.num_key_points - 1, 1)).to(self.device)), dim=2)
        else:
            input_2d = gt_2d

        gt_2d = gt_2d.detach()
        gt_X1 = gt_X1.detach()

        input_2d = input_2d.reshape(self.num_view*self.batch_size, -1)

        output_3d = self.net(input_2d)

        output_3d_v = []
        synt_R_v = []
        gt_2d_v = []
        gt_X1_v = []
        for i in range(self.num_view):
            output_3d_v.append(output_3d[(i*self.batch_size):((i+1)*self.batch_size), ...])
            synt_R_v.append(synt_rotations[(i*self.batch_size):((i+1)*self.batch_size), ...])
            gt_2d_v.append(gt_2d[(i*self.batch_size):((i+1)*self.batch_size), ...])
            gt_X1_v.append(gt_X1[(i*self.batch_size):((i+1)*self.batch_size), ...])

        for i in range(self.num_view):
            output_3d_i = output_3d_v[i].reshape(self.batch_size, self.num_key_points - 1, 3)

            if self.coefficients[1] > 0:
                loss1 = loss_no_scale(gt_X1_v[i], output_3d_i, num_keypoint=self.num_key_points - 1, keep_keypoint=True)
                loss += self.coefficients[1] * loss1
                if records is not None:
                    records[2] += loss1.mean().item() / float(self.print_frequence) / float(self.num_view)

            for j in range(self.num_view):
                R_relative_ij = torch.matmul(synt_R_v[j], torch.inverse(synt_R_v[i]))
                output_3d_ij = torch.transpose(torch.matmul(R_relative_ij, torch.transpose(output_3d_i, 1, 2)), 1, 2)
                if self.coefficients[2] > 0:
                    loss3 = loss_no_scale(gt_X1_v[j], output_3d_ij, num_keypoint=self.num_key_points - 1, keep_keypoint=True)
                    loss += self.coefficients[2] * loss3
                    if records is not None:
                        records[3] += loss3.mean().item() / float(self.print_frequence) / float(self.num_view)

                output_2d_ij = (output_3d_ij[:, :, :2]).reshape(self.batch_size, -1)
                if self.coefficients[0] > 0:
                    loss0 = loss_no_scale(gt_2d_v[j], output_2d_ij, num_keypoint=self.num_key_points - 1, keep_keypoint=True)
                    loss += self.coefficients[0] * loss0
                    if records is not None:
                        records[1] += loss0.mean().item()/float(self.print_frequence)/float(self.num_view**2)

        elementwise_loss = loss.clone().detach()

        loss = loss.mean()

        return loss, elementwise_loss

    def train_lifter(self, sample_per_epoch=None,):
        print(' ')
        print('-----------------------------------------------------------')
        print('Training network')
        print('-----------------------------------------------------------')
        print(' ')
        if not os.path.exists(self.weights_path + self.subfolder_path):
            os.makedirs(self.weights_path + self.subfolder_path)
        if sample_per_epoch is None:
            self.gt_2d_real_list = []
            for i in range(self.num_view):
                self.gt_2d_real_list.append([])
            for dataset in self.dataset:
                _, _, gt_2d_real_list = data_loader(dataset, self.data_path, multiview=self.num_view)
                print("We have ", len(gt_2d_real_list[0]), " training sample <"+str(self.num_view)+"-pairs> in", dataset, 'dataset')
                for i in range(self.num_view):
                    self.gt_2d_real_list[i] += gt_2d_real_list[i]

            self.num_train_samples = len(self.gt_2d_real_list[0])
        else:
            self.num_train_samples = sample_per_epoch
        print("We have ", self.num_train_samples, " training sample <"+str(self.num_view)+"-pairs> in total.")

        self.init_distribution()
        for i in range(self.initial_diffusion_steps):
            distribution_diffusion(self.target_dtb[..., 1:], dim=2, alpha=self.alpha_distribution_diffution)
            distribution_diffusion(self.root_target[..., 1:], dim=1, alpha=self.alpha_distribution_diffution)

        num_batch = self.num_train_samples//self.batch_size
        print("Begin training")
        for epoch in range(self.train_epoch):
            self.net.train()
            print("Begin training epoch " + str(epoch+1)+"/"+str(self.train_epoch))
            self.loss_record = [0, 0, 0, 0]
            for batch in range(num_batch):
                if self.coefficients[3] >= 0:
                    synt_extrinsic = torch.rand(self.num_view, 3) * 2 - 1
                    synt_extrinsic = torch.repeat_interleave(synt_extrinsic, repeats=self.batch_size, dim=0)
                else:
                    synt_extrinsic = torch.rand(self.num_view * self.batch_size, 3) * 2 - 1
                synt_extrinsic[:, 0] *= 0.05
                synt_extrinsic[:, 1] *= 3.142
                synt_extrinsic[:, 2] *= 0.05
                synt_extrinsic *= float(batch) / float(num_batch) / 2.0
                synt_extrinsic = synt_extrinsic.to(self.device)

                loss, elementwise_loss = self.epoch_train_synthetic(synt_extrinsic, self.loss_record)

                self.loss_record[0] += loss.item() / float(self.print_frequence)

                if (batch + 1) % self.print_frequence == 0:
                    print("batch " + str(batch + 1) + "/" + str(num_batch) + ": loss_lifter is " + str(loss.item()))
                    print("L_all, L_X2D, L_3D, L_X3D, L_reproj_ij are respectively", self.loss_record)

                    self.loss_record = [0, 0, 0, 0]

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.alpha_distribution_diffution > 0:
                    delta_elementwiseloss = torch.abs(elementwise_loss-self.last_elementwiseloss)
                    delta_elementwiseloss /= (10.0 * self.num_view)
                    delta_elementwiseloss = torch.pow(10.0, -delta_elementwiseloss)
                    delta_elementwiseloss = torch.cat((delta_elementwiseloss[:10],
                                                       torch.zeros(1).to(delta_elementwiseloss.device),
                                                       delta_elementwiseloss[10:]), dim=0)
                    self.last_elementwiseloss = elementwise_loss

                    distribution_diffusion(self.target_dtb[..., 1:], dim=2, alpha=self.alpha_distribution_diffution *
                                                                                  delta_elementwiseloss)
                    distribution_diffusion(self.root_target[..., 1:], dim=1, alpha=self.alpha_distribution_diffution)

            torch.save(self.net.state_dict(), self.weights_path + self.subfolder_path+'lifter_' + str(epoch + 1) + '.pth')

            print("End training epoch " + str(epoch+1)+"/"+str(self.train_epoch))
            self.lr_scheduler.step()
        print("Finish training")

    def eval_lifter(self, testset='h36m', eval_current=None):
        if not os.path.exists(self.weights_path + self.subfolder_path):
            os.makedirs(self.weights_path + self.subfolder_path)

        if testset == 'h36m':
            eval_view = self.num_view
        else:
            eval_view = 1
        _, _, self.val_gt, self.val_gt_d2 = data_loader(testset, self.data_path, multiview=eval_view)
        for i in range(eval_view):
            self.val_gt[i] = torch.cat(self.val_gt[i], dim=0)
            self.val_gt_d2[i] = torch.cat(self.val_gt_d2[i], dim=0)
        self.num_val_samples = len(self.val_gt_d2[0])
        print("We have ", self.num_val_samples, " evaluation samples for,", testset)

        if eval_current is None:
            epoch_eval = self.train_epoch
        else:
            epoch_eval = 1

        for epoch in range(epoch_eval):
            if eval_current is None:
                if os.path.exists(self.weights_path + self.subfolder_path + 'lifter_' + str(epoch + 1) + '.pth'):
                    self.net.load_state_dict(torch.load(self.weights_path + self.subfolder_path + 'lifter_' +
                                                        str(epoch + 1) + '.pth',map_location=self.device))
                    print("Loaded weights at:" + self.weights_path + self.subfolder_path + 'lifter_' + str(epoch + 1) + '.pth')
                else:
                    break
            self.net.eval()

            pjpe = torch.zeros(3)
            pck = torch.zeros(4)

            num_batch = self.num_val_samples // self.batch_size
            for batch in range(num_batch):
                gt, gt_d2 = [], []
                for i in range(eval_view):
                    gt.append(self.val_gt[i][batch * self.batch_size:((batch + 1) * self.batch_size), ...])
                    gt_d2.append(self.val_gt_d2[i][batch * self.batch_size:((batch + 1) * self.batch_size), ...])
                gt = torch.cat(gt, dim=0).to(self.device)
                gt_d2 = torch.cat(gt_d2, dim=0).to(self.device)

                input_2d = gt_d2.float().clone().detach()
                input_2d = ((input_2d - input_2d[:, 0:1, :])[:, 1:, :]).reshape(eval_view * self.batch_size, -1)
                forbenius_gt_d2 = forbenius_norm_multi_dim(input_2d)
                input_2d = input_2d / forbenius_gt_d2
                if self.use_indicator:
                    input_2d = torch.cat((input_2d.reshape(eval_view * self.batch_size, self.num_key_points - 1, 2),
                                       torch.ones(eval_view * self.batch_size, self.num_key_points - 1, 1).to(self.device)),
                                      dim=2).reshape(eval_view * self.batch_size, -1)

                gt_d3 = ((gt - gt[:, 0:1, :])[:, 1:, :]).reshape(eval_view * self.batch_size, -1)
                forbenius_gt_d3 = forbenius_norm_multi_dim(gt_d3)

                pose3d = self.net(input_2d)

                forbenius_pose3d = forbenius_norm_multi_dim(pose3d)
                reproject = pose3d.reshape(-1, self.num_key_points - 1, 3)
                reproject = torch.cat((torch.zeros(eval_view * self.batch_size, 1, 3).to(self.device), reproject), dim=1)

                reproject = reproject / forbenius_pose3d.unsqueeze(2) * forbenius_gt_d3.unsqueeze(2) + gt[:, 0:1, :]
                pjpe[0] += pjpe_loss(reproject, gt, align=True).item()

                pck[:2] += pck_count(reproject, gt, align=True)
                pck[2:] += pck_count(reproject, gt, align=True, threshold=150)

                if (batch + 1) % 100 == 0:
                    print(batch + 1, "/", num_batch)
            pjpe = pjpe / float(num_batch)
            pjpe[1] = pck[..., 0] / pck[..., 1] * 100
            pjpe[2] = pck[..., 2] / pck[..., 3] * 100

            if eval_current is None:
                print("The MPJPE loss for Epoch ", epoch+1, "is ", pjpe[0].item(), "mm with pck half head is",
                      pjpe[1].item(), "%, pck 150mm is", pjpe[2].item(), "%")
            else:
                print("The MPJPE loss for Epoch ", eval_current, "is ", pjpe[0].item(), "mm with pck half head is",
                      pjpe[1].item(), "%, pck 150mm is", pjpe[2].item(), "%")

    def draw_inferenced_image(self, num=1):
        print(' ')
        print('-----------------------------------------------------------')
        print('Drawing inference image')
        print('-----------------------------------------------------------')
        print(' ')
        if not os.path.exists('./examples/'):
            os.makedirs('./examples/')
        image_path, bbox, gt_2d_list = data_loader('coco', self.data_path, multiview=1)
        gt_2d_list = gt_2d_list[0]
        image_path = image_path[0]
        bbox = bbox[0]

        for i in range(num):
            print(image_path[i])
            img = Image.open(image_path[i])
            img.save('./examples/coco_' + str(i) + '_origin.png')
            img = img.crop((bbox[i][1].item(), bbox[i][0].item(), bbox[i][1].item()+bbox[i][3].item(),
                            bbox[i][0].item()+bbox[i][2].item()))
            img.save('./examples/coco_'+str(i)+'_cropped.png')

            input_real = gt_2d_list[i].unsqueeze(0).to(self.device)
            input_real = ((input_real - input_real[:, 0:1, :])[:, 1:, :2])
            input_real = input_real / forbenius_norm_multi_dim(input_real)
            if self.use_indicator:
                input_real = torch.cat((input_real.reshape(1, self.num_key_points - 1, 2),
                                        torch.ones(1, self.num_key_points - 1, 1).to(self.device)), dim=2).reshape(1, -1)

            pose3d = self.net(input_real)
            reproject = pose3d.reshape(1, self.num_key_points - 1, 3).cpu()
            reproject = torch.cat((torch.zeros(1, 1, 3), reproject), dim=1)
            draw_skeleton(reproject.detach(), plt_show=False, save_path='./examples/coco_'+str(i)+'_lifted.png',
                          background=img)
            print((i+1), '/', num)
        print('The inferenc images are saved inside ./examples')


if __name__ == "__main__":
    trainer = Trainer(opts, pretrain=True)

    if trainer.task == 'generation' or trainer.task == 'g':
        trainer.show_generation_sample()
    if trainer.task == 'train' or trainer.task == 't':
        trainer.train_lifter(sample_per_epoch=128)
    if trainer.task == 'inference' or trainer.task == 'i':
        trainer.draw_inferenced_image(4)
    if trainer.task == 'make-distribution' or trainer.task == 'md':
        trainer.compute_crafted_distribution()
    if trainer.task == 'all':
        trainer.compute_crafted_distribution()
        trainer.show_generation_sample()
        trainer.train_lifter(sample_per_epoch=128)
        trainer.draw_inferenced_image(4)
