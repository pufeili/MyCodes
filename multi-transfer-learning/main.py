import os
import time
import sys
import argparse
from solver import Solver

# 设置超参数
parser = argparse.ArgumentParser(description="Double transfer learning")

parser.add_argument("--batch_size", type=int, default=16)

parser.add_argument("--lr", type=float, default=0.0001, help="learning rate(default: 0.001)")
parser.add_argument("--max_epoch", type=int, default=20)
parser.add_argument('--optimizer', type=str, default='adam', metavar='N', help='which optimizer')   # SGD

parser.add_argument("--fusion_mode", type=str, default="sum", help="sum or none")
parser.add_argument('--save_epoch', type=int, default=30, metavar='N', help='when to restore the model')
parser.add_argument('--save_model', action='store_true', default=False, help='save_model or not')

parser.add_argument('--source', type=str, default='im', metavar='N', help='source dataset')     # im/bk
parser.add_argument('--target', type=str, default='bc', metavar='N', help='target dataset')     # bk/bc
parser.add_argument('--num_bc', type=int, default='3', help='which dataset of BC')     # bk/bc
args = parser.parse_args()


def main():
    solver = Solver(args, source=args.source, target=args.target, learning_rate=args.lr, batch_size=args.batch_size,
                    num_bc=args.num_bc, fuse_mode=args.fusion_mode, optimizer=args.optimizer)

    now_time = str(time.ctime())
    start_time = time.time()
    print("Date: %s" % now_time)
    # 创建record并将运行的结果保存在txt文件中，模型保存在checkpoints中
    record_num = 0
    record_train = './records/%s2%s_%s_num_data_%s_record_num_%s.txt' % (
        args.source, args.target, args.fusion_mode, args.num_bc, record_num)

    while os.path.exists(record_train):
        record_num += 1
        record_train = './records/%s2%s_%s_num_data_%s_record_num_%s.txt' % (
            args.source, args.target, args.fusion_mode, args.num_bc, record_num)

    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    if not os.path.exists('./records'):
        os.mkdir('./records')

    if record_train:
        with open(record_train, 'a') as log:
            print("Date: %s \n"
                  "source=%s, target=%s, lr=%f, batch_size=%d, max_epoch=%d, optimzer=%s\n"
                  % (now_time, args.source, args.target, args.lr, args.batch_size, args.max_epoch, args.optimizer),
                  file=log)

    for t in range(args.max_epoch):
        solver.train(t, record_file=record_train)
        if t % 1 == 0:
            solver.test(t, record_file=record_train, save_model=args.save_model)

    end_time = time.time()
    t = end_time - start_time
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    print("Total time: %.1fh %.1fmin %.1fs \n" % (h, m, s))
    with open(record_train, 'a') as f:
        print("Total time: %.1fh %.1fmin %.1fs \n" % (h, m, s), file=f)


if __name__ == '__main__':
    main()

