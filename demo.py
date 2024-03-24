from aro_net.Demo.Trainer.aro import demo as demo_train_aro
from aro_net.Demo.Trainer.mash import demo as demo_train_mash
from aro_net.Demo.Generator3D.aro import demo as demo_gen_aro
from aro_net.Demo.Generator3D.mash import demo as demo_gen_mash

if __name__ == "__main__":
    demo_train_aro()
    demo_train_mash()
    demo_gen_aro()
    demo_gen_mash()
