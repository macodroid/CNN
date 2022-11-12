import torch


def get_optimizer(optimizer_config: dict, model):
    if optimizer_config["name"] == "Adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config["learning_rate"],
            betas=(optimizer_config["beta1"], optimizer_config["beta2"]),
        )
    elif optimizer_config["name"] == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=optimizer_config["learning_rate"],
            nesterov=optimizer_config["nesterov"],
            momentum=optimizer_config["momentum"],
        )
    elif optimizer_config["name"] == "RMSprop":
        return torch.optim.RMSprop(
            model.parameters(),
            lr=optimizer_config["learning_rate"],
            momentum=optimizer_config["momentum"],
            alpha=optimizer_config["alpha"],
        )
