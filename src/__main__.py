#!/usr/bin/env python
# -*- coding: utf-8 -*-
from csci_utils import canvas
import argparse
from git import Repo
import functools
import os
import json
from pprint import pprint


def main():
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
    comments = dict(
        hexsha=repo.head.commit.hexsha[:8],
        submitted_from=repo.remotes.origin.url,
        dt=repo.head.commit.committed_datetime.isoformat(),
        branch=os.environ.get("GITHUB_ACTIVE_BRANCH", ""),
        use_late_days=os.environ.get("USE_LATE_DAYS", 0),
        is_dirty=repo.is_dirty(),
        docs="https://fast-krig.readthedocs.io/en/latest/index.html"
    )

    assignment = course.assignments["Final Project"]
    assignment.submit(
        dict(
            submission_type="online_url",
            url=url,
        ),
        comment=dict(
            text_comment=json.dumps(comments)
        ),
        **{},
    )


if __name__ == "__main__":
    main()
