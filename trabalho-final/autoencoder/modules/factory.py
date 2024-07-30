from .rede_conv_factory import ConvAutoencoderFactory
from .rede_factory import MLPAutoencoderFactory
from .abstract_rede_factory import AbstractRedeFactory
from .vae_factory import VAEFactory


class FactoryProducer:
    @staticmethod
    def getFactory(factory_type: str) -> AbstractRedeFactory:
        if factory_type == "ConvAutoencoder":
            return ConvAutoencoderFactory()
        elif factory_type == "MLPAutoencoder":
            return MLPAutoencoderFactory()
        elif factory_type == "VAE":
            return VAEFactory()
