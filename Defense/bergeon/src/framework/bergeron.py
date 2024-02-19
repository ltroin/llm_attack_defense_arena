import dataclasses

from universalmodels import ModelSrc

from src.framework.framework_model import FrameworkModel
from src.framework.primary import Primary
from src.framework.secondary import Secondary
from src.logger import root_logger


@dataclasses.dataclass
class DetectionReport:
    """Record of which components are activated during a bergeron model's generation"""

    dangerous_prompt: bool = False
    dangerous_response: bool = False
    sealed: bool = False

    def seal(self):
        """Marks a record as sealed to differentiate between bergeron models that seal it and primary models that ignore it"""

        self.sealed = True

    def __setattr__(self, key, value):
        if key == "sealed" and value is False and self.sealed is True:
            raise AttributeError("Cannot unseal a sealed report")
        if self.sealed:
            raise AttributeError("Cannot set the attribute of a report once it has been sealed")
        super().__setattr__(key, value)


class Bergeron(FrameworkModel):
    """The combined bergeron model.  The primary model responds to user input as usual.  The secondary model vets both the input and the response"""

    def __init__(self, primary_model: Primary, secondary_model: Secondary):
        """
        Args:
            primary_model: The primary framework model to use
            secondary_model: The secondary framework model to use"""

        self.primary = primary_model
        self.secondary = secondary_model

    @classmethod
    def from_model_names(cls, primary_model_name: str, secondary_model_name: str, rephrase_model_name: str = "dev/echo",):
        """Creates a bergeron model from the names of its primary and secondary models

        Args:
            primary_model_name: The name of the primary model
            secondary_model_name: The name of the secondary model
            rephrase_model_name: The name of the rephrasing model
            primary_model_src: The suggested source of the primary model to load. Defaults to AUTO
            secondary_model_src: The suggested source of the secondary model to load. Defaults to AUTO
            rephrase_model_src: The suggested source of the rephrasing model to load. Defaults to AUTO
        Returns:
            An instance of a bergeron model"""

        primary = Primary.from_model_name(primary_model_name, model_src=primary_model_src)
        secondary = Secondary.from_model_names(secondary_model_name, rephrase_model_name, secondary_model_src=secondary_model_src, rephrase_model_src=rephrase_model_src)

        return cls(primary, secondary)

    @property
    def name(self):
        return f"C({self.primary.model.name_or_path}, {self.secondary.critique_model.name_or_path})"

    def generate(self, promptobj: list,benign=False, detection_report: DetectionReport = None, params=None, **kwargs):
        """Generates a response to the prompt from the primary model while sing the secondary to monitor for unsafe text

        Args:
            prompt: The prompt to generate a response for
            detection_report: A detection report to use for recording which components have activated. Sealed after usage
        Returns:
            The generated safe response string"""
        prompt = promptobj[0]
        prompt = prompt.strip("\n").strip(" ")
        
        # primary_response = self.primary.generate(prompt, **kwargs)

        root_logger.debug("Rephrasing prompt...")
        rephrased_prompt = self.secondary.rephrase(prompt)

        root_logger.debug("Critiquing prompt...")
        input_critique = self.secondary.critique_prompt(rephrased_prompt, **kwargs)
        # input_critique = []
        print("input_critique",input_critique)

        # Checking the response for unsafe content and correcting
        if len(input_critique) > 0:
            if detection_report is not None:
                detection_report.dangerous_prompt = True

            root_logger.debug("Generating conscience...")
            sanitized = self.secondary.make_conscience_prompt(prompt, input_critique)
            print("sanitized",sanitized)
            root_logger.debug("Generating response with conscience...")
            if params:
                primary_response = self.primary.generate(sanitized,params, **kwargs)
            else:
                primary_response = self.primary.generate(sanitized, **kwargs)
        else:
            root_logger.debug("if unfiltered,use the original one")
            if benign:
                primary_response = self.primary.generate(sanitized, **kwargs)
            else:
                primary_response=promptobj[2]
            # if params:
            #     primary_response = self.primary.generate(sanitized,params, **kwargs)
            # else:
            #     primary_response = self.primary.generate(sanitized, **kwargs)

        root_logger.debug("Generating final response critique...")
        resp_critique = self.secondary.critique_response(primary_response, **kwargs)
        print("resp_critique",resp_critique)
        # Checking the response for unsafe content and correcting
        if len(resp_critique) > 0:
            if detection_report is not None:
                detection_report.dangerous_response = True

            root_logger.debug("Generating final correction...")
            correction_prompt = self.secondary.make_correction_prompt(primary_response, resp_critique)
            if params:
                primary_response = self.primary.generate(correction_prompt,params, **kwargs)
            else:
                primary_response = self.primary.generate(correction_prompt, **kwargs)

        # Seal the detection report so that it is clear that it was used
        if detection_report is not None:
            detection_report.seal()

        return primary_response
