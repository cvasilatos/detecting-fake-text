#!/usr/bin/env python

import argparse
import connexion
import os
from flask import send_from_directory, redirect
from flask_cors import CORS

from backend import AVAILABLE_MODELS

__author__ = 'Hendrik Strobelt, Sebastian Gehrmann'

CONFIG_FILE_NAME = 'lmf.yml'
projects = {}

app = connexion.App(__name__, debug=False)


class Project:
    def __init__(self, lm, config):
        self.config = config
        self.lm = lm(config)


def get_all_projects():
    res = {}
    for k in projects.keys():
        res[k] = projects[k].config
    return res


def analyze(analyze_request):
    project = analyze_request.get('project')
    text = analyze_request.get('text')

    res = {}
    if project in projects:
        p = projects[project]
        res = p.lm.check_probabilities(text, topk=20)

    return {
        "request": {'project': project, 'text': text},
        "result": res
    }


@app.route('/')
def redir():
    return redirect('client/index.html')


@app.route('/client/<path:path>')
def send_static(path):
    """ serves all files from ./client/ to ``/client/<path:path>``

    :param path: path from api call
    """
    return send_from_directory('client/dist/', path)


@app.route('/data/<path:path>')
def send_data(path):
    """ serves all files from the data dir to ``/data/<path:path>``

    :param path: path from api call
    """
    print(f"Got the data route for {path}")
    return send_from_directory(args.dir, path)


app.add_api('server.yaml')

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='gpt2')
parser.add_argument("--nodebug", default=True)
parser.add_argument("--address", default="0.0.0.0")
parser.add_argument("--port", default="5001")
parser.add_argument("--nocache", default=False)
parser.add_argument("--dir", type=str, default=os.path.abspath('data'))

parser.add_argument("--no_cors", action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()

    if not args.no_cors:
        CORS(app.app, headers='Content-Type')

    app.run(port=int(args.port), debug=not args.nodebug, host=args.address)
else:
    args, _ = parser.parse_known_args()
    try:
        model = AVAILABLE_MODELS[args.model]
    except KeyError:
        print(f"Model {args.model} not found. Make sure to register it. Loading GPT-2 instead.")
        model = AVAILABLE_MODELS['gpt2']

    projects['gpt-2-small'] = Project(model, args.model)
