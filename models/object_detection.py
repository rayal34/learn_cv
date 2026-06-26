import torch
import torch.nn as nn


class ObjectDetectionFromResnet(nn.Module):
    def __init__(self, backbone, num_classes, model_config):
        super().__init__()
        backbone_fc_features_in = backbone.fc.in_features
        # remove the last layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        self.flatten = nn.Flatten()

        # add the classfication head
        fc_features_in = backbone_fc_features_in
        clf_fc_layers = []
        if model_config.clf_fc_layers:
            for spec in model_config.clf_fc_layers:
                clf_fc_layers.extend(
                    [
                        nn.Linear(fc_features_in, out_features=spec.out_features),
                        nn.ReLU(),
                    ]
                )
                fc_features_in = spec.out_features

        clf_fc_layers.append(nn.Linear(fc_features_in, out_features=num_classes))
        self.clf_fc_layers = nn.Sequential(*clf_fc_layers)

        # add the bounding box head
        fc_features_in = backbone_fc_features_in
        bbox_fc_layers = []
        if model_config.bbox_fc_layers:
            for spec in model_config.bbox_fc_layers:
                bbox_fc_layers.extend(
                    [
                        nn.Linear(fc_features_in, out_features=spec.out_features),
                        nn.ReLU(),
                    ]
                )
            fc_features_in = spec.out_features

        # doesn't directly predict the bounding box coordinates.  Instead predicts (batch, cx, cy, w', h') where
        # cx, cy are the center coordinates and w = exp(w') and h = exp(h')
        bbox_fc_layers.append(nn.Linear(fc_features_in, out_features=4))
        self.bbox_fc_layers = nn.Sequential(*bbox_fc_layers)

    def forward(self, x):
        out = self.backbone(x)
        out = self.flatten(out)

        clf_out = self.clf_fc_layers(out)

        box_out = self.bbox_fc_layers(out)  # shape = (batch, cx, cy, w', h')
        centers = torch.sigmoid(box_out[:, [0, 1]])
        w_h = torch.sigmoid(torch.exp(box_out[:, -2:]))

        x1 = centers[:, 0] - w_h[:, 0] / 2
        y1 = centers[:, 1] - w_h[:, 1] / 2
        x2 = centers[:, 0] + w_h[:, 0] / 2
        y2 = centers[:, 1] + w_h[:, 1] / 2
        bbox_out = torch.stack([x1, y1, x2, y2], dim=1)
        return clf_out, bbox_out
