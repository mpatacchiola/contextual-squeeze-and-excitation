import torch
import torchvision
import numpy as np

class UpperCaSE():

    def __init__(self, backbone, device, tot_iterations, start_lr=0.00025, stop_lr=1e-6, transform_fn=None):
        self.backbone = backbone
        self.device = device
        self.tot_iterations = tot_iterations
        self.start_lr = start_lr
        self.stop_lr = stop_lr
        self.transform_fn = transform_fn
        self.parameters_values_list = list()
        self.head = None

        # Accumulates the params of the adapters
        adaptive_params_list = list()
        for name, param in backbone.named_parameters():
            if("gamma_generator" in name):
                adaptive_params_list.append(param)

        if(len(adaptive_params_list) > 0):
            self.optimizer = torch.optim.Adam(adaptive_params_list, lr=start_lr)
        else:
            print("[WARNING] Parameters list is empty for optimizer")
            quit()

           
    def predict(self, context_images, context_labels, target_images, verbose=None):
        tot_classes = torch.max(context_labels).item() + 1
        tot_context_images = context_images.shape[0]
        nll = torch.nn.NLLLoss(reduction='mean')

        # Forward over the context data with CaSE in adaptive mode
        with torch.no_grad():
            self.backbone.set_mode(adapter="train", backbone="eval") # adaptive mode
            context_embeddings = self.backbone(context_images.to(self.device))
            self.backbone.set_mode(adapter="eval", backbone="eval") # inference mode

        # Define a linear head
        tot_embeddings = context_embeddings.shape[1]   
        self.head = torch.nn.Linear(in_features=tot_embeddings, 
                                    out_features=tot_classes, 
                                    bias=True, 
                                    device=self.device)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)             
        optimizer_head = torch.optim.Adam(self.head.parameters(), lr=0.001, weight_decay= 1e-5)

        # Optimize the parameters of the linear head using context data
        batch_size=128
        lr_linspace = np.linspace(start=0.001, stop=1e-5, num=self.tot_iterations, endpoint=True)
        self.head.train()
        for iteration in range(self.tot_iterations):
            # Sample a mini-batch
            indices = np.random.choice(tot_context_images, size=batch_size, replace=True) # replace to deal with small tasks
            if(self.transform_fn is not None):
                with torch.no_grad():
                    inputs = self.backbone(self.transform_fn(context_images[indices].to(self.device)))
            else:
                inputs = context_embeddings[indices]
            labels = context_labels[indices]
            # Set the learning rate
            lr = lr_linspace[iteration]
            for param_group in optimizer_head.param_groups: param_group["lr"] = lr
            # Optimization
            optimizer_head.zero_grad()
            log_probs = torch.log_softmax(self.head(inputs), dim=1)
            loss = nll(log_probs, labels)
            loss.backward()
            optimizer_head.step()
        # Free memory
        del context_embeddings

        # Estimate the logits for the target images
        with torch.no_grad():
            self.head.eval()
            self.backbone.set_mode(adapter="eval", backbone="eval") # inference mode
            logits = self.head(self.backbone(target_images.to(self.device)))
            return torch.log_softmax(logits, dim=1) # Return log-probs


    def predict_batch(self, context_images, context_labels, target_images, reset, verbose=None):
        if(reset==True):
            tot_classes = torch.max(context_labels).item() + 1
            tot_context_images = context_images.shape[0]
            nll = torch.nn.NLLLoss(reduction='mean')
            # Compute the context embeddings on CPU (only once per dataset)
            with torch.no_grad():
                    context_images = context_images.to("cpu")
                    self.backbone = self.backbone.to("cpu")
                    self.backbone.set_mode(adapter="train", backbone="eval")
                    context_embeddings = self.backbone(context_images)
                    tot_embeddings = context_embeddings.shape[1]
                    self.backbone.set_mode(adapter="eval", backbone="eval")
                    self.backbone = self.backbone.to(self.device)

            self.head = torch.nn.Linear(in_features=tot_embeddings, 
                                    out_features=tot_classes, 
                                    bias=True, 
                                    device=self.device)
            torch.nn.init.zeros_(self.head.weight)
            torch.nn.init.zeros_(self.head.bias)             
            optimizer_head = torch.optim.Adam(self.head.parameters(), lr=0.001, weight_decay= 1e-5)
        
            batch_size=128
            splits=1
            lr_linspace = np.linspace(start=0.001, stop=1e-5, num=self.tot_iterations, endpoint=True)
            self.head.train()
            for iteration in range(self.tot_iterations):
                indices = np.random.choice(tot_context_images, size=batch_size, replace=True)
                if(self.transform_fn is not None):
                     with torch.no_grad():
                         selected_context_images = context_images[indices]
                         selected_context_images = selected_context_images.to(self.device)
                         inputs = self.backbone(self.transform_fn(selected_context_images))
                else:
                     inputs = context_embeddings[indices]
                labels = context_labels[indices]
                log_probs = torch.log_softmax(self.head(inputs.to(self.device)), dim=1)
                loss = nll(log_probs, labels)
                loss.backward()
                # Set the learning rate
                lr = lr_linspace[iteration]
                for param_group in optimizer_head.param_groups: param_group["lr"] = lr
                # Optim step                
                optimizer_head.step()
                optimizer_head.zero_grad()
                
            # Free memory
            del context_embeddings
            self.backbone = self.backbone.to(self.device)

        # Estimate the logits for the target images
        if(target_images is not None):
            with torch.no_grad():
                self.head.eval()
                self.backbone.set_mode(adapter="eval", backbone="eval")
                logits = self.head(self.backbone(target_images.to(self.device))) 
                return torch.log_softmax(logits, dim=1)

                    
    def learn(self, task_idx, tot_tasks, context_images, context_labels, target_images, target_labels, verbose=None):
        tot_classes = torch.max(context_labels).item() + 1
        tot_context_images = context_images.shape[0]
        nll = torch.nn.NLLLoss(reduction='mean')

        # Forward over the context data with CaSE in adaptive mode
        with torch.no_grad():
            self.backbone.set_mode(adapter="train", backbone="eval") # adaptive mode
            context_embeddings = self.backbone(context_images.to(self.device))
        
        # Define a linear head
        tot_embeddings = context_embeddings.shape[1]
        self.head = torch.nn.Linear(in_features=tot_embeddings, 
                                    out_features=tot_classes, 
                                    bias=True, 
                                    device=self.device)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)  
        optimizer_head = torch.optim.Adam(self.head.parameters(), lr=0.001, weight_decay= 1e-5)
        
        # Optimize the parameters of the linear head using context data
        batch_size=128
        lr_linspace = np.linspace(start=0.001, stop=1e-5, num=self.tot_iterations, endpoint=True)
        self.head.train()
        for iteration in range(self.tot_iterations):
            # Sample a mini-batch
            indices = np.random.choice(tot_context_images, size=batch_size, replace=True)
            if(self.transform_fn is not None):
                inputs = self.backbone(self.transform_fn(context_images[indices].to(self.device)))
            else:
                inputs = context_embeddings[indices]
            labels = context_labels[indices]
            # Set the learning rate
            lr = lr_linspace[iteration]
            for param_group in optimizer_head.param_groups: param_group["lr"] = lr
            # Optimization
            optimizer_head.zero_grad()
            log_probs = torch.log_softmax(self.head(inputs), dim=1)
            loss = nll(log_probs, labels)
            loss.backward()
            optimizer_head.step()
        # Free memory
        del context_embeddings
        
        # Optimize the CaSE parameters
        self.head.eval()
        self.backbone.set_mode(adapter="train", backbone="eval") # adaptive mode
        all_images = torch.cat([context_images, target_images], dim=0)
        all_labels = torch.cat([context_labels, target_labels], dim=0)
        tot_images = all_images.shape[0]
        self.head.zero_grad()
        batch_size=128
        tot_iterations = max(1, tot_images//batch_size)
        for iteration in range(tot_iterations):
            indices = np.random.choice(tot_images, size=batch_size, replace=True) 
            inputs = all_images[indices]
            labels = all_labels[indices]
            logits = self.head(self.backbone(inputs))
            loss = nll(torch.log_softmax(logits, dim=1), labels)
            loss.backward()

        # Backprop every 16 tasks
        if(task_idx%16==0 and task_idx>0):
            # Set learning rate
            lr_linspace = np.linspace(start=self.start_lr, stop=self.stop_lr, num=tot_tasks, endpoint=True)
            for param_group in self.optimizer.param_groups: param_group["lr"] = lr_linspace[task_idx]
            # Optim step
            self.optimizer.step()
            # Zero the gradients
            self.backbone.zero_grad()
            self.optimizer.zero_grad()
            print(f"Optimizer step; lr: {lr_linspace[task_idx]:.8f}")

            
        # Estimate the logits for the target images
        with torch.no_grad():
            self.head.eval()
            self.backbone.set_mode(adapter="train", backbone="eval")
            self.backbone(context_images.to(self.device))
            self.backbone.set_mode(adapter="eval", backbone="eval")
            logits = self.head(self.backbone(target_images.to(self.device)))
        return torch.log_softmax(logits, dim=1) # Return log-probs
                    
