import argparse
import os

os.getcwd()
from train_rbd import train

import networks


name = "binding"

parser = argparse.ArgumentParser()

#=======================================================================================================================
parser.add_argument(
        "--model_name",
        "-mn",
        help="network for training."
        "-mn for ",
        type=str,
        default="XBCR_net",
        required=False,
    )
parser.add_argument(
        "--data_name",
        "-dn",
        help="data name for training."
        "-dn for ",
        type=str,
        default=name,
        required=False,
    )
parser.add_argument(
    "--type",
    help="Training type, full or rbd or multi",
    # default="full",
    default="rbd",
    type=str,
    required=False,
)
parser.add_argument(
    "--model_num",
    help="The model number.",
    type=int,
    default=0,
)
parser.add_argument(
    "--max_epochs",
    help="The maximum number of epochs, -1 means following configuration.",
    type=int,
    default=1000,
)
parser.add_argument(
    "--include_light",
    help="include light or not.",
    type=int,
    default=1,
)
#=======================================================================================================================
args = parser.parse_args()
#=======================================================================================================================

model_num=args.model_num

batch_size=12
nb_epochs1 = args.max_epochs

model_name=args.model_name
data_name=args.data_name
include_light=args.include_light

# network setting
net_core = networks.get_net(model_name)

os.getcwd()
print(os.getcwd())
model_path=os.path.join('.','models',data_name,data_name+'-'+model_name,'model')
data_path=os.path.join('.','data',data_name)
print('model:',model_path,'  data:',data_path)
print(os.path.abspath(data_path))

# training data
pos_path = os.path.join(data_path,'exper')
neg_path = os.path.join(data_path,'nonexp')

train(net_core=net_core, model_path=model_path,model_num=model_num,include_light=include_light, pos_path=pos_path, neg_path=neg_path, batch_size=batch_size, nb_epochs1=nb_epochs1)
