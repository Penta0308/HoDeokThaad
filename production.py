from urllib import parse
import flask
from flask import request

import modules

webserver = flask.Flask("HoDeokThaad")
webserver.config['JSON_AS_ASCII'] = False

@webserver.route("/")
def flask_eval():
    resp = {}
    try:
        t = modules.tokenizer(parse.unquote(request.args['t'], encoding='UTF-8'))
    except KeyError:
        resp['code'] = 400
        return flask.jsonify(resp)

    if len(t) == 0:
        resp['code'] = 500
        return flask.jsonify(resp)

    eval_data = data.Dataset(
        [data.Example.fromlist([t, "0"], fields=[('text', TEXT), ('label', LABEL)]), ],
        fields={'text': TEXT, 'label': LABEL})

    eval_iterator = data.BucketIterator.splits(
        (eval_data,),
        batch_size=1,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        device=device)[0]

    resp['code'] = 200
    resp['data'] = {'token': t}

    with torch.no_grad():
        for batch in eval_iterator:
            resp['data']['rate'] = 1.0 - float(torch.sigmoid(model(batch.text).squeeze(1)[0]))
            return flask.jsonify(resp)


webserver.run(host="0.0.0.0", port=modules.config_get("port"))
