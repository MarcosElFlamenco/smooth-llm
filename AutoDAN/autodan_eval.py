from utils.eval_utils import build_arg_parser, run_autodan_eval


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_autodan_eval(args, attack_mode=args.attack_mode)