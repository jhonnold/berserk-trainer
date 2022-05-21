# Copied from the SF NNUE repository and modified for use with Berserk networks
# https://github.com/glinscott/nnue-pytorch/blob/master/run_games.py

import re
import os
import subprocess
import sys
import time
import argparse
import shutil


def find_networks(root_dir):
    """ Find the set of nets that are available for testing, going through the full subtree """

    p = re.compile("nn-epoch[0-9]*.nnue")

    networks = []
    for path, _, files in os.walk(root_dir, followlinks=False):
        for filename in files:
            if p.match(filename):
                epoch_num = re.search(r'epoch(\d+)', filename)
                if int(epoch_num.group(1)) % 5 == 0:
                    networks.append(os.path.join(path, filename))

    return networks


def parse_ordo(root_dir, networks):
    """ Parse an ordo output file for rating and error """

    ordo_file_name = os.path.join(root_dir, "ordo.out")
    ordo_scores = {}

    for name in networks:
        ordo_scores[name] = (-500, 1000)

    if not os.path.exists(ordo_file_name):
        return ordo_scores

    with open(ordo_file_name, "r") as ordo_file:
        lines = ordo_file.readlines()
        network_lines = filter(lambda x: "nn-epoch" in x, lines)

        for l in network_lines:
            fields = l.split()
            ordo_scores[fields[1]] = (float(fields[3]), float(fields[4]))

    return ordo_scores


def run_match(best, root_dir, c_chess_exe, concurrency, book_file_name,
              berserk_base, berserk_test):
    """ Run a match using c-chess-cli adding pgns to a file to be analysed with ordo """

    pgn_file_name = os.path.join(root_dir, "out.pgn")
    command = "{} -each nodes=20000 option.Hash=8 option.Threads=1 -gauntlet -games 200 -rounds 1 -concurrency {}".format(
        c_chess_exe, concurrency)
    command = (
        command +
        " -openings file={} order=random -repeat -resign count=3 score=700 -draw count=8 score=10"
        .format(book_file_name))
    command = command + " -engine cmd={} name=master".format(berserk_base)

    for net in best:
        command = command + " -engine cmd={} name={} option.EvalFile={}".format(
            berserk_test, net, os.path.join(os.getcwd(), net))

    command = command + " -pgn {} 0 2>&1".format(pgn_file_name)

    print("Running match with c-chess-cli ... {}".format(pgn_file_name),
          flush=True)

    c_chess_out = open(os.path.join(root_dir, "c_chess.out"), 'w')
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)

    seen = {}
    for line in process.stdout:
        line = line.decode('utf-8')
        c_chess_out.write(line)

        if 'Score' in line:
            epoch_num = re.search(r'epoch(\d+)', line)
            if epoch_num.group(1) not in seen:
                sys.stdout.write('\n')
            seen[epoch_num.group(1)] = True
            sys.stdout.write('\r' + line.rstrip())

    sys.stdout.write('\n')
    c_chess_out.close()

    if process.wait() != 0:
        print("Error running match!")


def run_ordo(root_dir, ordo_exe, concurrency):
    """ run an ordo calcuation on an existing pgn file """

    pgn_file_name = os.path.join(root_dir, "out.pgn")
    ordo_file_name = os.path.join(root_dir, "ordo.out")

    command = "{} -q -G -J  -p  {} -a 0.0 --anchor=master --draw-auto --white-auto -s 100 --cpus={} -o {}".format(
        ordo_exe, pgn_file_name, concurrency, ordo_file_name)

    print("Running ordo ranking ... {}".format(ordo_file_name), flush=True)

    ret = os.system(command)
    if ret != 0:
        print("Error running ordo!")


def run_round(
    root_dir,
    explore_factor,
    ordo_exe,
    c_chess_exe,
    berserk_base,
    berserk_test,
    book_file_name,
    concurrency,
):
    """ run a round of games, finding existing nets, analyze an ordo file to pick most suitable ones, run a round, and run ordo """

    # find a list of networks to test
    networks = find_networks(root_dir)
    if len(networks) == 0:
        print("No .nnue files found in {}".format(root_dir))
        time.sleep(10)
        return
    else:
        print("Found {} nn-epoch*.nnue files".format(len(networks)))

    # Get info from ordo data if that is around
    ordo_scores = parse_ordo(root_dir, networks)

    # provide the top 3 nets
    print("Best nets so far:")
    ordo_scores = dict(
        sorted(ordo_scores.items(), key=lambda item: item[1][0], reverse=True))
    count = 0
    for net in ordo_scores:
        print("   {} : {} +- {}".format(net, ordo_scores[net][0],
                                        ordo_scores[net][1]))
        count += 1
        if count == 3:
            break

    # get top 3 with error bar added, for further investigation
    print("Measuring nets:")
    ordo_scores = dict(
        sorted(
            ordo_scores.items(),
            key=lambda item: item[1][0] + explore_factor * item[1][1],
            reverse=True,
        ))
    best = []
    for net in ordo_scores:
        print("   {} : {} +- {}".format(net, ordo_scores[net][0],
                                        ordo_scores[net][1]))
        best.append(net)
        if len(best) == 3:
            break

    # run these nets against master...
    run_match(best, root_dir, c_chess_exe, concurrency, book_file_name,
              berserk_base, berserk_test)

    # and run a new ordo ranking for the games played so far
    run_ordo(root_dir, ordo_exe, concurrency)


def main():
    # basic setup
    parser = argparse.ArgumentParser(
        description="Finds the strongest .nnue in tree, playing games.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="""The directory where to look, recursively, for .nn.
                 This directory will be used to store additional files,
                 in particular the ranking (ordo.out)
                 and game results (out.pgn and c_chess.out).""",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Number of concurrently running threads",
    )
    parser.add_argument(
        "--explore_factor",
        default=1.5,
        type=float,
        help=
        "`expected_improvement = rating + explore_factor * error` is used to pick select the next networks to run.",
    )
    parser.add_argument(
        "--ordo_exe",
        type=str,
        default=".\\ordo.exe",
        help="Path to ordo, see https://github.com/michiguel/Ordo",
    )
    parser.add_argument(
        "--c_chess_exe",
        type=str,
        default=".\\c-chess-cli.exe",
        help="Path to c-chess-cli, see https://github.com/lucasart/c-chess-cli",
    )
    parser.add_argument(
        "--berserk_base",
        type=str,
        default=".\\berserk.exe",
        help=
        "Path to berserk master (reference version), see https://github.com/jhonnold/berserk",
    )
    parser.add_argument(
        "--berserk_test",
        type=str,
        help=
        "(optional) Path to new berserk binary, if not set, will use berserk_base",
    )
    parser.add_argument(
        "--book_file_name",
        type=str,
        default="4moves_noob.epd",
        help=
        "Path to a suitable book, see https://github.com/official-stockfish/books",
    )
    args = parser.parse_args()

    berserk_base = args.berserk_base
    berserk_test = args.berserk_test
    if berserk_test is None:
        berserk_test = berserk_base

    if not shutil.which(berserk_base):
        sys.exit("berserk base is not executable !")

    if not shutil.which(berserk_test):
        sys.exit("berserk test is not executable!")

    if not shutil.which(args.ordo_exe):
        sys.exit("ordo is not executable!")

    if not shutil.which(args.c_chess_exe):
        sys.exit("c_chess_cli is not executable!")

    if not os.path.exists(args.book_file_name):
        sys.exit("book does not exist!")

    while True:
        run_round(
            args.root_dir,
            args.explore_factor,
            args.ordo_exe,
            args.c_chess_exe,
            berserk_base,
            berserk_test,
            args.book_file_name,
            args.concurrency,
        )


if __name__ == "__main__":
    main()
