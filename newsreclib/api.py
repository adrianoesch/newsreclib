import hydra, pyrootutils, torch
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from newsreclib import utils
log = utils.get_pylogger(__name__)

from flask import Flask, request

server = Flask(__name__)
model = None

@server.route('/encode/user',methods=['POST'])
def user_encoder():
    batch = get_batch_from_request()
    with torch.no_grad():
        user_vector = model.encode_user(batch)
    return user_vector

@server.route('/encode/news',methods=['POST'])
def news_encoder():
    batch = get_batch_from_request()
    with torch.no_grad():
        news_vector = model.encode_news(batch)
    return news_vector

def get_batch_from_request():
    requestData = request.data.json()
    batch = [{}]
    return batch

def get_model_class_from_path(model_path):
    import importlib
    module = importlib.import_module(".".join(model_path.split('.')[:-1]))
    return getattr(module,model_path.split('.')[-1])


    
@utils.task_wrapper
def api(cfg):
    global model

    log.info("Load model from checkpoints...")
    ckpt = torch.load(cfg.ckpts_path,map_location=torch.device('cpu'))
    if cfg.hyper_param_path_replace:
        print('replace')
        for k,v in ckpt['hyper_parameters'].items():
            if k.endswith("_path"):
                ckpt['hyper_parameters'][k] = v.replace(
                    cfg.hyper_param_path_replace.pattern,
                    cfg.hyper_param_path_replace.replacement
                )

    model_class = get_model_class_from_path(cfg.model._target_)
    model = model_class(**ckpt['hyper_parameters'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    log.info("Run server...")
    server.run(debug=True,port=8008)


@hydra.main(version_base="1.3", config_path="../configs", config_name="api.yaml")
def main(cfg):
    api(cfg)
    
if __name__ == "__main__":
    main()