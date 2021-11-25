#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import collections
import logging
from transformers import BertConfig

from model.model import BertForDistantRE
from utils.train_utils import *
from utils.utils import idx_to_rel

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def flatten_trips_dict(d):
    flat_dict = {}
    for dic in d:
        flat_dict[dic] = d[dic]
    return flat_dict


def long_tail_split(all, upper_20=False):
    """
    Returns rare triples if upper_20 == False and common triples if upper_20 == True
    """
    with open(config.lower_half_trips, "rb") as wf:
        lower_80_ids = pickle.load(wf)

    lower_80_results = collections.defaultdict(list)
    kept, total = 0, 0
    index_to_rel_types = idx_to_rel(config.relations_file_types)
    rel_to_idx_names = read_relations(config.relations_file)
    logger.info('SIZE OF LOWER 80 SET: {}'.format(len(lower_80_ids)))
    logger.info('SIZE OF ALL DICT: {}'.format(len(all['labels'])))
    logger.info('SIZE OF RESULTS: {}'.format(len(all)))

    for label, logit, group, pred, name, trip, in zip(all['labels'], all['logits'], all['groups'], all['preds'],
                                                      all['names'], all['trips']):
        total += 1
        h, r, t = trip.split('\t')
        rel_type = index_to_rel_types[r]
        rel_name_idx = str(rel_to_idx_names[rel_type])
        trip_id = '\t'.join([h, rel_name_idx, t])
        # print('TRIP TEXT: {}'.format(trip_id))
        # print('LOWER 80 SAMPLE: {}'.format(list(lower_80_ids)[0]))
        if upper_20:
            if trip_id not in lower_80_ids:  # NOT = UPPER
                kept += 1
                lower_80_results['labels'].append(label)
                lower_80_results['logits'].append(logit)
                lower_80_results['groups'].append(group)  # groups
                lower_80_results['preds'].append(pred)
                lower_80_results['names'].append(name)
                lower_80_results['trips'].append(trip)
        else:
            if trip_id in lower_80_ids:  # LOWER
                kept += 1
                lower_80_results['labels'].append(label)
                lower_80_results['logits'].append(logit)
                lower_80_results['groups'].append(group)  # groups
                lower_80_results['preds'].append(pred)
                lower_80_results['names'].append(name)
                lower_80_results['trips'].append(trip)
    return lower_80_results, kept, total


def evaluate_test(model, model_dir, set_type="test",
                  load_eval: bool = False,
                  run_label: str = ''):
    eval_dataset = load_dataset(set_type, logger, ent_types=True)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=config.eval_batch_size)

    # Eval
    logger.info("***** Running evaluation {} *****".format(set_type))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", config.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    result_tracker = collections.defaultdict(list)
    eval_logits, eval_labels, eval_preds, eval_groups, eval_names, eval_trips = [], [], [], [], [], []
    trip_sent_count = collections.defaultdict(int)

    if load_eval:
        # Load evaluation raw data
        fname = os.path.join(model_dir, set_type + "_raw_eval_data.pkl")
        logger.info("Using model: {}".format(model_dir))
        logger.info("Loading raw results file: {}".format(fname))
        with open(fname, "rb") as wf:
            eval = pickle.load(wf)
        eval_loss = eval['loss']
        total_trips = 0
        for label, logit, pred, group, names in zip(eval['labels'], eval['logits'], eval['preds'], eval['groups'],
                                                    eval['names']):
            r = label.item()
            one_trip_per_bag = set()
            for name in names:
                h, t = name[0].item(), name[1].item()
                trip = "\t".join([str(h).lower(), str(r).lower(), str(t).lower()])
                # Eval distinct trip names in each eval group to better compare to eval names experiment
                # Results 1 pos/neg per distinct trip name
                if trip not in one_trip_per_bag:
                    one_trip_per_bag.add(trip)
                    total_trips += 1
                    trip_sent_count[trip] += 1
                    result_tracker['labels'].append(label)
                    result_tracker['logits'].append(logit)

                    # Each group value is [2]
                    result_tracker['groups'].append(group)  # groups

                    result_tracker['preds'].append(pred)
                    result_tracker['names'].append(name)
                    result_tracker['trips'].append(trip)
        kept_trips = total_trips

        logger.info("Kept {} trips of {} total trips. Percent: {}%.".format(kept_trips, total_trips,
                                                                            (kept_trips / total_trips)))
        logger.info("Length of labels: {}.".format(len(result_tracker['labels'])))

        # print('BBBBBB', 'labels', torch.stack(result_tracker['labels']).shape)
        # print('BBBBBB', 'logits', torch.stack(result_tracker['logits']).shape)
        # print('BBBBBB', 'groups', torch.stack(result_tracker['groups']).shape)

        eval = {
            'loss': eval_loss,
            'labels': torch.stack(result_tracker['labels']), # B, [36756]
            'logits': torch.stack(result_tracker['logits']), # B x C, [36756, 394]
            'groups': torch.stack(result_tracker['groups']), # B x 2, [36756, 2]
            'preds': np.asarray(result_tracker['preds']),
            'names': torch.stack(result_tracker['names'])
        }
    else:
        # If not loading saved results, run model inference
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(config.device) for t in batch)

            with torch.inference_mode():
                # print('XXX', batch[0].shape, batch[1].shape, batch[2].shape, batch[4].shape)

                inputs = {
                    "input_ids": batch[0], # [1, 16, 128], which is [B, G, MAX_LEN]
                    "entity_ids": batch[1], # [1, 16, 128]
                    "attention_mask": batch[2], # [1, 16, 128]
                    "labels": batch[4], # [1]
                    "is_train": False
                }
                tmp_eval_loss, logits = model(**inputs)
                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1

            # trip_names is [1, 16, 2], which is [B, G, 2] -- I think this is the "groups"
            # It looks like the following:
            # trip_names    torch.Size([1, 16, 2])
            # tensor([[[7080, 29974],
            #          [3943, 15662],
            #          [3943, 13144],
            #          [3943, 13144], [..]]]

            trip_names = batch[5].detach().cpu()

            unique_trips_in_group = set()

            # Then for each element in the batch -- trip is [16, 2]
            for trip in trip_names:

                # If this [16, 2] tensor does not appear in unique_trips_in_group
                if trip not in unique_trips_in_group:

                    # Add this [16, 2] matrix to unique_trips_in_group
                    unique_trips_in_group.add(trip)

                    # Add the labels and logits to eval_labels and eval_logits
                    eval_labels.append(inputs["labels"].detach().cpu())
                    eval_logits.append(logits.detach().cpu())

                    # print('batch[3] is', batch[3].shape) # -> [1, 2]
                    eval_groups.append(batch[3].detach().cpu())  # groups, each element is [B, 2] so [1, 2]

                    eval_names.append(trip_names)  # names, trip_names is [1, 16, 2]
                    eval_preds.append(torch.argmax(logits.detach().cpu(), dim=1).item())

        del model, batch, logits, tmp_eval_loss, eval_dataloader, eval_dataset  # memory mgmt

        eval = {
            'loss': eval_loss / nb_eval_steps,
            'labels': torch.cat(eval_labels),
            'logits': torch.cat(eval_logits),
            'preds': np.asarray(eval_preds),
            'groups': torch.cat(eval_groups), # This is a list of [1, 2] tensors which leads to [N, 2]
            'names': torch.cat(eval_names)
        }

        # print('eval[groups] is', eval['groups'].shape)

        # import sys
        # sys.exit(0)

    # Get all positive relationship lables
    rel2idx = read_relations(config.relations_file_types)
    pos_rel_idxs = list(rel2idx.values())
    rel_idx_na = rel2idx['na']
    del pos_rel_idxs[rel_idx_na]

    a = accuracy_score(eval['labels'].numpy(), eval['preds'])
    p, r, f1, support = precision_recall_fscore_support(eval['labels'].numpy(), eval['preds'], average='micro',
                                                        labels=pos_rel_idxs)
    logger.info('Accuracy (including "NA"): {}\nP: {}, R: {}, F1: {}'.format(a, p, r, f1))
    results = {}
    results['new_results'] = {
        'acc_with_na': a,
        'scikit_precision': p,
        'scikit_recall': r,
        'scikit_f1': f1,
        "loss": eval_loss,
        "counter": eval['labels'].shape
    }

    # What's the shape of eval['groups'] here ? It's [36756, 2]
    results['original'] = compute_metrics(eval['logits'], eval['labels'], eval['groups'], set_type, logger,
                                          ent_types=True)
    results["loss"] = eval_loss
    logger.info("Results: %s", results)

    if load_eval:
        # Save evaluation results
        with open(os.path.join(model_dir, set_type + "_metrics_from_load_{}.txt".format(run_label)), "w") as wf:
            json.dump(results, wf, indent=4)

    else:
        # Save evaluation results
        with open(os.path.join(model_dir, set_type + "_metrics.txt"), "w") as wf:
            json.dump(results, wf, indent=4)

        # Save evaluation raw data
        with open(os.path.join(model_dir, set_type + "_raw_eval_data.pkl"), "wb") as wf:
            pickle.dump(eval, wf)


def main(argv):
    # Get number of relations
    num_labels = len(read_relations(config.relations_file_types))

    # Load model
    model_dir = argv[0] # '[insert model dir here]'
    logger.info("Evaluate the checkpoint: %s", model_dir)
    model = BertForDistantRE(BertConfig.from_pretrained(model_dir), num_labels,
                             rel_embedding=config.rel_embedding, bag_attn=False)
    model.load_state_dict(torch.load(model_dir + "/pytorch_model.bin", map_location=torch.device(config.device)))
    model.to(config.device)

    # Load raw results (don't re-run model inference)
    load_eval = False
    # XXX LEAVE THIS ON "FALSE"

    # Run full set, lower 80 and upper 20
    runs = ['FULL_SET']
    for run_label in runs:

        # Evaluation
        evaluate_test(model, model_dir, "test", load_eval=load_eval, run_label=run_label)


if __name__ == "__main__":
    main(sys.argv[1:])
