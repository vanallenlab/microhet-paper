class wsi_lrnr():
  def __init__(self, label_col) -> None:
      hparams = {'num_ettes' : 10,
            'num_replicas' : 1000,
            'ratio' : 1,
            'non_lin' : 'None',
            'iter' : 0,
            'log_weight_decay' : 0}
      
      self.label_col = label_col

      run = wandb.init(config = hparams)
      config = wandb.config

      random_state = config.iter
      random.seed(a=random_state, version=2)
      seed_everything(random_state)

      self.non_lin = non_lin_choices[config.non_lin]
      #Read the dataset
      self.svs_path = pd.read_csv('svs_pathology.csv')

      # define ordinal encoding
      encoder = OrdinalEncoder()
      # transform data
      svs_path[label_col] = encoder.fit_transform(svs_path[[label_col]])

      # Convert labels to torch tensors so you can train
      svs_path[label_col] = torch.from_numpy(svs_path[label_col].values.astype(float)).type(torch.LongTensor) 

      gss = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=random_state)

      #Split patients into training and testing
      train_idx, val_idx = list(gss.split(X = list(range(len(svs_path))), groups = svs_path['npy_name']))[0]

      train_path, val_path = svs_path.iloc[train_idx], svs_path.iloc[val_idx]

      #Upsample the minority classes, create num_replicas etc.
      train_path = balance_labels_upsample(train_path, label_col, random_state = random_state, num_replicas = config.num_replicas, ratio = config.ratio)
      val_path = balance_labels_upsample(val_path, label_col, random_state = random_state, num_replicas = config.num_replicas, ratio = config.ratio)

      assert set(train_path['npy_name']).intersection(val_path['npy_name']) == set()

      #Create Train Dataset
      train_dataset = WSIV(
              paths=train_path.npy_name.values,
              slide_ids=train_path.svs_name.values,
              labels=train_path[label_col].values,
              transform_compose=RGBTrainTransform(full_size = 512, crop_size = 512, s = 0),
              num_ettes = config.num_ettes,
              non_lin = non_lin
          )

      #Create Eval Dataset
      val_dataset = WSIV(
              paths=val_path.npy_name.values,
              slide_ids=val_path.svs_name.values,
              labels=val_path[label_col].values,
              transform_compose=RGBTrainTransform(full_size = 512, crop_size = 512, s = 0),
              num_ettes = config.num_ettes,
              non_lin = non_lin
          )

      #Create Train Data Loader
      train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=1)

      #Create Train Data Loader
      val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=1)

      #Create model
      model = PretrainedResnet50FT(hparams = {'lr' : 1e-5, 
                                              'num_classes' : 2,
                                              'weight_decay' : 10 ** config.log_weight_decay})

      #Create a logger
      logger = WandbLogger(name = 'wsiv', save_dir = '.', project = 'wsiv')

      early_stop_callback = EarlyStopping(
        monitor='val_f1',
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='max'
      )

      #Create a trainer obj
      trainer = pl.Trainer(gpus=-1, 
                          max_epochs=15, 
                          callbacks=[early_stop_callback], 
                          fast_dev_run = 0,
                          logger = logger,
                          log_every_n_steps=1,
      )

      #Fit the trainer and start training
      trainer.fit(model, train_dataloader, val_dataloader)

class resnet_lrnr():
  def __init__(self, hparams) -> None:
  # {'num_replicas' : 1,
  #             'ratio' : 1,
  #             'label' : 'E',
  #             'non_lin' : 'relu',
  #             'iter' : 0,
  #             'log_weight_decay' : -5,
  #             'log_lr' : -5,
  #             'dropout' : 0.2,
  #             'crop_size' : 512,
  #             'batch_size' : 32,
  #             'accumulate_grad_batches' : 128
  # }

    self.hparams = hparams 
    
    self.num_replicas = num_replicas
    self.ratio = ratio
    self.label = label
    self.non_lin = non_lin
    self.iter = iter
    self.log_weight_decay = log_weight_decay
    self.log_lr = log_lr
    self.dropout = dropout
    self.crop_size = crop_size
    self.batch_size = batch_size
    self.accumulate_grad_batches = accumulate_grad_batches
    self.csv_path = csv_path #'tile_path_mut.csv'

    self.run = wandb.init(config = hparams)
    config = wandb.config

    random_state = self.iter
    random.seed(a=random_state, version=2)
    seed_everything(random_state)

    non_lin = non_lin_choices[config.non_lin]

    self.read_data()

  def read_data(self):
    #Read the dataset
    self.data = pd.read_csv(self.csv_path)

  def pre_process_data(self):
    # define ordinal encoding
    self.encoder = OrdinalEncoder()
    # transform data
    self.data[self.label] = self.encoder.fit_transform(self.data[[self.label]])

    # Convert labels to torch tensors so you can train
    self.data[self.label] = torch.from_numpy(self.data[self.label].values.astype(float)).type(torch.LongTensor) 
  
  def train_test_split(self):

    self.gss = GroupShuffleSplit(n_splits=int(100), train_size=0.7, random_state=random_state)

    td_ctr = collections.Counter([])
    vd_ctr = collections.Counter([])
    num_labels = len(set(self.data[self.label]))

    splits = list(self.gss.split(X = list(range(len(self.data))), groups = self.data['slide_id']))
    while (len(td_ctr) != num_labels)  or (len(vd_ctr) != num_labels):
        
        #Split patients into training and testing
        train_idx, val_idx = next(iter(splits))

        train_path, val_path = self.data.iloc[train_idx], self.data.iloc[val_idx]

        td_ctr = collections.Counter(train_path[self.label].values)
        vd_ctr = collections.Counter(val_path[self.label].values)

        # print(td_ctr, vd_ctr)

        train_path = train_path.sample(frac = 1, replace = False, random_state = self.random_state)
        val_path = val_path.sample(frac = 1, replace = False, random_state = self.random_state)

    #Upsample to 50 50 
    self.train_path = balance_labels_upsample(train_path, self.label, random_state = self.random_state, num_replicas = self.num_replicas, ratio = self.ratio)
    self.val_path = balance_labels_upsample(val_path, self.label, random_state = self.random_state, num_replicas = self.num_replicas, ratio = self.ratio)

    assert set(self.train_path['slide_id']).intersection(self.val_path['slide_id']) == set()

  def create_datasets(self):
    #Create Train Dataset
    self.train_dataset = SlideDataset(
            paths=self.train_path.full_path.values,
            slide_ids=self.train_path.slide_id.values,
            labels=self.train_path[self.label].values,
            transform_compose=RGBTrainTransform(full_size = 512, crop_size = config.crop_size, s = 1),
        )

    #Create Eval Dataset
    self.val_dataset = SlideDataset(
            paths=self.val_path.full_path.values,
            slide_ids=self.val_path.slide_id.values,
            labels=self.val_path[self.label].values,
            transform_compose=RGBEvalTransform(full_size = 512, crop_size = config.crop_size),
        )
  def create_dataloaders(self):
    #Create Train Data Loader
    self.train_dataloader = self.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=8)

    #Create Train Data Loader
    self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=8)

  def create_model(self):
    #Create model
    self.model = PretrainedResnet50FT(hparams = {'lr' : 10 ** config.log_lr, 
                                            'num_classes' : 2, 
                                            'weight_decay' : 10 ** config.log_weight_decay,
                                            'dropout' : config.dropout 
                                          })

    #Create a logger
    logger = WandbLogger(name = 'wandb_logger', save_dir = '.', project = 'PROFILE')

    early_stop_callback = EarlyStopping(
      monitor='val_loss',
      min_delta=0.00,
      patience=4,
      verbose=False,
      mode='min'
    )

    #Create a trainer obj
    self.trainer = pl.Trainer(gpus=-1, 
                        max_steps=5000, 
                        callbacks=[], 
                        fast_dev_run=0,
                        logger = logger,
                        val_check_interval=0.5,
                        log_every_n_steps=10,
                        limit_train_batches=200,
                        limit_val_batches=20,
                        accumulate_grad_batches=self.accumulate_grad_batches
                        # overfit_batches=10
    )

  def fit(self):
    #Fit the trainer and start training
    self.trainer.fit(self.model, self.train_dataloader, self.val_dataloader)

