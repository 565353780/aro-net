from aro_net.Demo.trainer import demo as demo_train
from aro_net.Demo.Generator3D.aro import demo as demo_gen_aro
from aro_net.Demo.Generator3D.mash import demo as demo_gen_mash

if __name__ == "__main__":
    demo_train()
    demo_gen_aro()
    demo_gen_mash()
