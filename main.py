import argparse
from prompt_process import load_json
import attack
import defence
import download_models

def run_attack(model_name, attack_type):
    """
    Execute an attack on a specified model using the chosen attack type.
    """
    if attack_type == "AutoDAN":
        print(f"Applying AutoDAN attack to {model_name}")
        attack_instance = attack.AutoDAN(model=model_name)
        attack_instance.run()
    elif attack_type == "GPTFuzz":
        print(f"Applying GPTFuzz attack to {model_name}")
        attack_instance = attack.GPTFuzz(model=model_name)
        attack_instance.run()
    elif attack_type == "DeepInception":
        print(f"Applying DeepInception attack to {model_name}")
        attack_instance = attack.DeepInception(model=model_name)
        attack_instance.run()
    elif attack_type == "Tap":
        print(f"Applying Tap attack to {model_name}")
        attack_instance = attack.Tap(model=model_name)
        attack_instance.run()
    elif attack_type == "Pair":
        print (f"Applying Pair attack to {model_name}")
        attack_instance = attack.Pair(model=model_name)
        attack_instance.run()
    elif attack_type == "Jailbroken":
        print(f"Applying Jailbroken attack to {model_name}")
        attack_instance = attack.Jailbroken(model=model_name)
        attack_instance.run()
    elif attack_type == "TemplateJailbreak":
        print (f"Applying TemplateJailbreak attack to {model_name}")
        attack_instance = attack.TemplateJailbreak(model=model_name)
        attack_instance.run()
    elif attack_type == "Parameters":
        print(f"Applying Parameters attack to {model_name}")
        attack_instance = attack.Parameters(model=model_name)
        attack_instance.run()
    elif attack_type == "GCG":
        print(f"Applying GCG attack to {model_name}")
        attack_instance = attack.GCG(model=model_name)
        attack_instance.run()
    else:
        print("Attack type not recognized.")

def run_defense(model_name, defense_type):
    """
    Apply a defense mechanism to a specified model using the chosen defense type.
    Please Note: If model_name is expected to certain defense, the file_path should be about that model.
    The reason is you can not check modelA response with modelB to see if the defense is working.
    """
    file_path = f"../../Results/{model_name}/Merged_{model_name}.json"
    if defense_type == "RALLM":
        print(f"Applying RALLM defense to {model_name} using file {file_path}")
        defence.RALLM(model=model_name, file=file_path).run()
    elif defense_type == "Baseline":
        
        print(f"Applying Baseline defense to {model_name} using file {file_path}")
        defence.Baseline(model=model_name, file=file_path).run()
    elif defense_type == "Aegis":
        
        print(f"Applying Aegis defense to {model_name} using file {file_path}")
        defence.Aegis(file=file_path).run()
    elif defense_type == "LLMGuard":
        
        print(f"Applying LLMGuard defense to {model_name} using file {file_path}")
        defence.LLMGuard(file=file_path).run()
    elif defense_type == "Smooth":
        print(f"Applying Smooth defense to {model_name} using file {file_path}")
        defence.Smooth(model=model_name, file=file_path).run()
    elif defense_type == "Moderation":
        print(f"Applying Moderation defense to {model_name} using file {file_path}")
        defence.Moderation(file_path=file_path).run()
    elif defense_type == "BergeonMethod":
        print(f"Applying BergeonMethod defense to {model_name} using file {file_path}")
        defence.BergeonMethod(file_path=file_path).run()
    else:
        print("Defense type not recognized.")

def main():
    parser = argparse.ArgumentParser(description="Run attack and defense mechanisms on AI models")
    parser.add_argument('--model', choices=['gpt-3.5-turbo', 'llama', 'vicuna','vicuna13','mistral'], required=False, help='Model to attack or defend')
    parser.add_argument('--mode', choices=['attack', 'defense','process'], required=False, help='Whether to run an attack or apply a defense or process the results.')
    parser.add_argument('--type', required=False, help='Type of attack or defense to run')
    parser.add_argument('--need-download', required=False,default="false", help='do you need to download the model?')
    args = parser.parse_args()

    if "false" in args.need_download:
        args.need_download = False
    else:
        args.need_download = True

    if args.need_download:
        print("downloading the model")            
        download_models.download(args.model)

    if args.mode == 'attack':
        run_attack(args.model, args.type)
    elif args.mode == 'defense':
        run_defense(args.model, args.type)
    elif args.mode == 'process':
        ##NOTE This must happen before the defense is applied
        load_json(f'./Results/{args.model}')

if __name__ == "__main__":
    main()
