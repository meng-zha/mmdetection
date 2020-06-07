from argparse import ArgumentParser
import os

from mmdet.apis import inference_detector, init_detector, show_result_pyplot, save_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    img_name = os.path.basename(args.img)
    work_path = os.path.dirname(args.checkpoint)

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    # show the results
    save_result_pyplot(model, args.img, result,os.path.join(work_path,img_name), score_thr=args.score_thr)


if __name__ == '__main__':
    main()
