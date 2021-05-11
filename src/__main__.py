#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pset_5.tasks import ByDayOfWeek, ByStars, ByYear
from csci_utils import canvas
from luigi import build
import argparse
from git import Repo
import functools
import os
import json
from pprint import pprint


parser = argparse.ArgumentParser()
parser.add_argument(
    "--full",
    action="store_true",
    help="Passing this argument performs analysis on all the data",
)
opt = parser.parse_args()


@functools.lru_cache(maxsize=5)
def load_df(task):
    return task(subset=not opt.full).get_results()


def answer_by(key, task=None):
    df = load_df(task)
    key = key.split("_")[-1]
    df.index = df.index.astype(str)
    return df.loc[key][0]


def main():
    build(
        [
            ByDayOfWeek(subset=not opt.full),
            ByStars(subset=not opt.full),
            ByYear(subset=not opt.full),
        ],
        local_scheduler=True,
    )

    repo = Repo(".")  # Get the current repo
    is_clean = not repo.is_dirty()
    commit_hash = repo.head.commit.hexsha[:8]
    # Initialize the canvas course object with the course ID and token.
    course = canvas.Course(
        course_id=os.environ["CANVAS_COURSE_ID"],
        canvas_token=os.environ["CANVAS_TOKEN"],
    )
    course.get_assignments_and_quizes()

    # find the url for the repo
    url = "https://github.com/tim-a-davis/{}/commit/{}".format(
        os.path.basename(repo.working_dir), repo.head.commit.hexsha
    )
    # Assign answers to questions

    assignment = course.assignments["Final Project"]
    assignment.submit(
        dict(
            submission_type="online_url",
            url=url,
        ),
        comment=dict(
            text_comment=json.dumps(course.get_submission_comments(submission))
        ),
        **{},
    )


if __name__ == "__main__":
    main()
