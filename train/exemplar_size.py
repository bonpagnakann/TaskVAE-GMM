import os

def calculate_exemp_size(dataset, new_class, person):
    holdout_sizes = {
        'hhar': {
            ('3', '2'): {0: [100, 394, 1550, 6100, 95], 1: [100, 393, 1547, 6085, 95], 2: [100, 390, 1520, 5928, 95]},
            ('22', '21', '31'): {0: [100, 394, 1550, 6100, 135], 1: [100, 393, 1547, 6085, 135], 2: [100, 390, 1520, 5928, 135]},
            '111': {0: [100, 394, 1550, 6100, 175], 1: [100, 393, 1547, 6085, 175], 2: [100, 390, 1520, 5928, 175]},
            '1111': {0: [100, 394, 1550, 6100, 215], 1: [100, 393, 1547, 6085, 215], 2: [100, 390, 1520, 5928, 215]}
        },
        'motion': {
            ('3', '2'): {0: [100, 197, 390, 770, 95], 1: [100, 195, 379, 737, 95], 2: [100, 191, 366, 699, 95]},
            ('22', '21', '31'): {0: [100, 197, 390, 770, 135], 1: [100, 195, 379, 737, 135], 2: [100, 191, 366, 699, 135]},
            '111': {0: [100, 197, 390, 770, 175], 1: [100, 195, 379, 737, 175], 2: [100, 191, 366, 699, 175]},
            '1111': {0: [100, 197, 390, 770, 215], 1: [100, 195, 379, 737, 215], 2: [100, 191, 366, 699, 215]}
        },
        'uci': {
            ('3', '2'): {0: [100, 142, 201, 286, 95], 1: [100, 142, 201, 285, 95], 2: [100, 140, 196, 274, 95]},
            ('22', '21', '31'): {0: [100, 142, 201, 286, 135], 1: [100, 142, 201, 285, 135], 2: [100, 140, 196, 274, 135]},
            '111': {0: [100, 142, 201, 286, 175], 1: [100, 142, 201, 285, 175], 2: [100, 140, 196, 274, 175]},
            '1111': {0: [100, 142, 201, 286, 215], 1: [100, 142, 201, 285, 215], 2: [100, 140, 196, 274, 215]}
        }, 
        'realworld': {
            ('3'): {0: [100, 303, 921, 2795, 95], 1: [100, 300, 898, 2692, 95], 2: [100, 299, 894, 2674, 95]},
            ('32','22'): {0: [100, 303, 921, 2795, 135], 1: [100, 300, 898, 2692, 135], 2: [100, 299, 894, 2674, 135]},
            ('211', '222'): {0: [100, 303, 921, 2795, 175], 1: [100, 300, 898, 2692, 175], 2: [100, 299, 894, 2674, 175]},
            ('1111', '2111', '3111'): {0: [100, 303, 921, 2795, 215], 1: [100, 898, 2692, 215], 2: [100, 299, 894, 2674, 215]},
        },
        'pamap': {
            '4': {0: [100, 240, 575, 1378, 95], 1: [100, 237, 562, 1332, 95], 2: [100, 235, 554, 1303, 95]},
            ('32','33'): {0: [100, 240, 575, 1378, 135], 1: [100, 237, 562, 1332, 135], 2: [100, 235, 554, 1303, 135]},
            ('321', '341'): {0: [100, 575, 1378, 175], 1: [100, 237, 562, 1332, 175], 2: [100, 235, 554, 1303, 175]},
            '2222': {0: [100, 240, 575, 1378, 215], 1: [100, 237, 562, 1332, 215], 2: [100, 235, 554, 1303, 215]},
            '11111': {0: [100, 240, 575, 1378, 255], 1: [100, 237, 562, 1332, 255], 2: [100, 235, 554, 1303, 255]},
        }
    }

    dataset_info = holdout_sizes.get(dataset, {})
    for classes, sizes_by_person in dataset_info.items():
        if new_class in classes if isinstance(classes, tuple) else new_class == classes:
            return sizes_by_person.get(person, None)
    return None

def get_output_file_paths(args, OUT_PATH, holdout_size):
    """
    Generates output file paths based on method and exemplar type.
    """

    # Define base folder structure
    base_path = f"{OUT_PATH}{args.dataset}/{args.base_classes}{args.new_classes}/Person_{args.person}"

    # Determine folder name based on method or exemplar type
    if args.exemplar == 'taskvae':
        folder_name = f"{base_path}/VAE_{args.vae_lat_sampling}_{args.latent_vec_filter}/log"

        filename_suffix = f"{args.person}_{args.method}_{args.exemplar}_{args.vae_lat_sampling}_{args.latent_vec_filter}_{holdout_size}"
        filename_suffix_t = f"t_{args.person}_{args.method}_{args.exemplar}_{args.vae_lat_sampling}_{args.latent_vec_filter}"

    elif args.exemplar == 'ddgr':
        folder_name = f"{base_path}/DDGR/log"

        filename_suffix = f"{args.person}_{args.method}_{args.exemplar}_{holdout_size}"
        filename_suffix_t = f"t_{args.person}_{args.method}_{args.exemplar}"

    elif args.exemplar == 'fetril':
        folder_name = f"{base_path}/FeTrIL/log"

        filename_suffix = f"{args.person}_{args.method}_{args.exemplar}_{holdout_size}"
        filename_suffix_t = f"t_{args.person}_{args.method}_{args.exemplar}"

    elif args.exemplar == 'taskvae_ratio':
        folder_name = f"{base_path}_{args.number}/log"

        filename_suffix = f"{args.person}_{args.method}_{args.exemplar}_{args.vae_lat_sampling}_{args.latent_vec_filter}_{holdout_size}"
        filename_suffix_t = f"t_{args.person}_{args.method}_{args.exemplar}_{args.vae_lat_sampling}_{args.latent_vec_filter}"


    elif args.method == 'kd_kldiv':
        folder_name = f"{base_path}/iCaRL_{args.number}/log"

        filename_suffix = f"{args.person}_{args.method}_{args.exemplar}_{str(args.lamda_old)}_{holdout_size}"
        filename_suffix_t = f"t_{args.person}_{args.method}_{args.exemplar}_{str(args.lamda_old)}"

    elif args.method == 'cn_lfc_mr':
        folder_name = f"{base_path}/LUCIR_{args.number}/log"

        filename_suffix = f"{args.person}_{args.method}_{args.exemplar}_{str(args.lamda_base)}_{holdout_size}"
        filename_suffix_t = f"t_{args.person}_{args.method}_{args.exemplar}_{str(args.lamda_base)}"

    elif args.method == 'ce_ewc':
        folder_name = f"{base_path}/EWC_Replay_{args.number}/log"

        filename_suffix = f"{args.person}_{args.method}_{args.exemplar}_{holdout_size}"
        filename_suffix_t = f"t_{args.person}_{args.method}_{args.exemplar}"

    else:
        folder_name = f"{base_path}/Random_{args.number}/log"

        filename_suffix = f"{args.person}_{args.method}_{args.exemplar}_{holdout_size}"
        filename_suffix_t = f"t_{args.person}_{args.method}_{args.exemplar}"

    # Create folders for log and statistics
    log_folder = os.path.join(folder_name, "log")
    timelog_folder = os.path.join(folder_name, "time_log")
    os.makedirs(log_folder, exist_ok=True)
    os.makedirs(timelog_folder, exist_ok=True)

    # Define output file paths
    outfile = f"{log_folder}/{filename_suffix}.txt"
    outfile_t = f"{timelog_folder}/{filename_suffix_t}.txt"

    return outfile, outfile_t
