from enum import Enum

import torch
import torch.distributed as dist

from helper.log import Log

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(args, name, fmt=':f', summary_type=Summary.AVERAGE):
        args.name = name
        args.fmt = fmt
        args.summary_type = summary_type
        args.reset()

    def reset(args):
        args.val = 0
        args.avg = 0
        args.sum = 0
        args.count = 0

    def update(args, val, n=1):
        args.val = val
        args.sum += val * n
        args.count += n
        args.avg = args.sum / args.count

    def all_reduce(args):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        total = torch.tensor([args.sum, args.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        args.sum, args.count = total.tolist()
        args.avg = args.sum / args.count

    def __str__(args):
        fmtstr = '{name} {val' + args.fmt + '} ({avg' + args.fmt + '})'
        return fmtstr.format(**args.__dict__)
    
    def summary(args):
        fmtstr = ''
        if args.summary_type is Summary.NONE:
            fmtstr = ''
        elif args.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif args.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif args.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % args.summary_type)
        
        return fmtstr.format(**args.__dict__)

class ProgressMeter(object):
    def __init__(args, num_batches, meters, prefix=""):
        args.batch_fmtstr = args._get_batch_fmtstr(num_batches)
        args.meters = meters
        args.prefix = prefix

    def display(args, batch):
        entries = [args.prefix + args.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in args.meters]
        Log.log('\t'.join(entries))
        
    def display_summary(args):
        entries = [" *"]
        entries += [meter.summary() for meter in args.meters]
        Log.log(' '.join(entries))

    def _get_batch_fmtstr(args, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
