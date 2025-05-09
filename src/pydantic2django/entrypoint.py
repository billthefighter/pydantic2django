from dataclasses import dataclass

from pydantic2django.dataclass.discovery import DataclassDiscovery

@dataclass
class ModuleConverter:
    """
    Convert a module to a set of Django models.
    """
    module: Module
    output_path: str

    def discover(self):
        """
        Discover the models in the module.
        """
        self.dataclass_discovery = DataclassDiscovery()



    def generate(self):
