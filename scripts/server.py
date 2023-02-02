#!/usr/bin/env python
# encoding: utf-8

from flask import Flask, request
from flask_wtf.csrf import CSRFProtect
import perplexity
from ppl import Perplexity

app = Flask('ai-gen-check')

csrf = CSRFProtect(app)
app.secret_key = 'dev'


@app.after_request
def add_cors(rv):
    rv.headers.add('Access-Control-Allow-Origin', 'localhost')
    rv.headers.add('Access-Control-Allow-Headers', 'X-CSRFToken')
    rv.headers.add('Access-Control-Allow-Credentials', 'true')
    return rv


@app.route('/api/perplexity', methods=['PUT'])
def get_perplexity():
    print(f"###### {request.json}")
    ppl_value = perplexity.perplexity(request.json['value'])
    ppl_object = Perplexity(ppl_value)
    print(f"Reply: {ppl_object.to_json()}")
    return ppl_object.to_json()


app.run(port=8000, debug=True)
