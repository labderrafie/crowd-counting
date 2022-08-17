import torch
from C_CNN import CCNN
from dsnet import DenseScaleNet, DDCB
from model import CANNet
import torch_pruning as tp

trained_model_path = '/home/l_abderrafie/Videtics/crowd counting/crowd-counting/models/DSNet/v2/model_best.pth.tar'
# trained_model_path = '/home/l_abderrafie/Videtics/crowd counting/crowd-counting/models/c_cnn/e200_cropv2/model_best.pth.tar'
# trained_model_path = '/home/l_abderrafie/Videtics/crowd counting/crowd-counting/models/cannet/v3/model_best.pth.tar'
def get_model():
    model = DenseScaleNet()
    model.load_state_dict(torch.load(trained_model_path, map_location=torch.device('cpu'))['state_dict'],)
    model.eval()
    return model

model = get_model()

#print(model)
# Global metrics
dummy_input = torch.randn(1, 3, 540, 960)
imp = tp.importance.MagnitudeImportance(p=2)
ignored_layers = []
for m in model.modules():
    if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
        ignored_layers.append(m)

total_steps = 1
pruner = tp.pruner.LocalMagnitudePruner(
    model,
    dummy_input,
    importance=imp,
    total_steps=total_steps,
    ch_sparsity=0.1,
    ignored_layers=ignored_layers,
)

for i in range(total_steps):
    ori_size = tp.utils.count_params(model)
    pruner.step()
    print(
        "  Params: %.2i  => %.2i "
        % (ori_size , tp.utils.count_params(model))
    )
# # save pruned model 
# torch.save(model, 'models/c_cnn/pruned_ccnn_new.pth.tar')
# print('model saved.')

with torch.no_grad():
    #print(model)
    print(model(dummy_input).shape)

# saving pruned model to ONNX while it's loaded :')
print('saving ONNX...')
torch.onnx.export(model, dummy_input, 'CCNN_list.onnx', input_names = ['input'], 
                  output_names = ['output'], opset_version=13, verbose = True)
print('ONNX saved!')
