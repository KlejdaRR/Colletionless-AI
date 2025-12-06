import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from Models.Teacher import Teacher
from Models.Student import Student

class CollaborativeClassroom:
    def __init__(self, num_students=8, moving_avg_window=100, eval_interval=200, val_batch_size=100):
        self.num_students = num_students  # How many neural networks in the classroom
        self.moving_avg_window = moving_avg_window  # How many recent losses to track per student
        self.eval_interval = eval_interval  # How often to find the "best student"
        self.val_batch_size = val_batch_size  # Size of validation batches for best student selection

        self.students = [Student() for _ in range(num_students)]
        # Creating num_students independent neural networks
        # Each student has different random initial weights -> Diverse starting points for learning

        self.optimizers = [optim.Adam(student.parameters(), lr=0.001) for student in self.students]
        # One optimizer per student: Each student learns at their own pace
        # Adam optimizer: Adaptive learning rate (smarter than basic SGD)
        # Learning rate 0.001

        self.teacher = Teacher()
        # Single teacher for all students
        # Starts with 1000 teaching opportunities
        # Will retire after providing 1000 correct answers (true labels)

        self.loss_memory = [deque(maxlen=moving_avg_window) for _ in range(num_students)]
        # Creating one loss history queue per student
        # deque(maxlen=100): Fixed-size queue that automatically discards old losses
        # Purpose: Tracking recent performance to identify the best student
        # Example:
        # Student 1: [0.45, 0.42, 0.38, ..., 0.12] (last 100 losses)

        self.best_student_idx = 0
        self.phase = "teacher_phase"
        # best_student_idx = 0: Starting by assuming student #0 is best (will update later)
        # phase = "teacher_phase": Starting with teacher-led learning

        # Virtualization: Storing validation batches for consistent evaluation
        self.validation_batches = []
        # Same data used every time we evaluate who's the best

        self.history = {
            'teacher_accuracy': [],
            'best_student_accuracy': [],
            'class_avg_accuracy': [],
            'class_surpassed_best': False,
            'step_when_surpassed': None
        }
        # Tracking: Accuracy over time for analysis
        # Milestone tracking: When class beats best student
        # Purpose: For plots and proving collective intelligence emerging

    def setup_validation_batches(self, val_loader, num_batches=5):
        # Extracting 5 batches from the validation dataset
        # and storing fixed validation batches for consistent evaluation
        self.validation_batches = []
        val_iter = iter(val_loader)

        for _ in range(num_batches):
            try:
                x_val, y_val = next(val_iter)
                self.validation_batches.append((x_val, y_val))
            except StopIteration:
                break

        print(f"Created {len(self.validation_batches)} validation batches for best student selection")

    def evaluate_student_on_validation(self, student_idx):
        ## Evaluating a single student on all validation batches
        student = self.students[student_idx]
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for x_val, y_val in self.validation_batches:
                # Example explaining a batch of 4 images
                output = student(x_val)  # shape: [4, 10] - 4 images, 10 classes each

                # Step 1: Getting predictions
                pred = output.argmax(dim=1, keepdim=True)
                # pred = [[1], [0], [7], [3]]

                # Step 2: Comparing with true labels
                # y_val = [1, 0, 7, 9]
                comparison = pred.eq(y_val.view_as(pred))
                # comparison = [[True], [True], [True], [False]]

                # Step 3: Counting correct predictions
                correct = comparison.sum().item()  # 3

                # Step 4: Updating totals
                total_correct += 3
                total_samples += 4  # len(x_val) = 4

        return total_correct / total_samples if total_samples > 0 else 0

    def find_best_student(self):
        # Finding the best student based on validation performance
        # Using validation accuracy instead of training loss because training loss can be misleading due to overfitting
        # Validation accuracy shows true generalization ability

        if not self.validation_batches:
            # Fallback to training loss if no validation batches available
            avg_losses = [self.get_moving_average_loss(i) for i in range(self.num_students)]
            return np.argmin(avg_losses)

        # Evaluating all students on validation set
        val_accuracies = []
        for i in range(self.num_students):
            accuracy = self.evaluate_student_on_validation(i)
            val_accuracies.append(accuracy)

        # Returning student with the highest validation accuracy
        return np.argmax(val_accuracies)

    # Checks if student has learning history: len(self.loss_memory[student_idx]) == 0
    # If no history: Returns float('inf') (infinity) - worst possible score
    # If it has history: Calculates average of recent losses
    # Student #3's loss_memory: [0.45, 0.42, 0.38, 0.35, 0.32]
    # get_moving_average_loss(3) → (0.45+0.42+0.38+0.35+0.32)/5 = 0.384
    def get_moving_average_loss(self, student_idx):
        if len(self.loss_memory[student_idx]) == 0:
            return float('inf')
        return np.mean(list(self.loss_memory[student_idx]))

    def evaluate_students(self, test_loader, step):
        teacher_correct = 0
        best_student_correct = 0
        class_total_correct = 0
        total_samples = 0

        with torch.no_grad():  # Disabling gradient calculation for efficiency
            for data, target in test_loader:  # Processing one batch of test data

                # Teacher Accuracy (if available):
                if self.teacher.calls_remaining > 0:
                    # Best student makes predictions on test batch
                    teacher_pred = self.students[self.best_student_idx].forward(data)

                    # Converting probabilities to class labels (0-9) with argmax
                    teacher_pred = teacher_pred.argmax(dim=1, keepdim=True)

                    # Comparing predictions with true labels (target) and counting how many are correct
                    teacher_correct += teacher_pred.eq(target.view_as(teacher_pred)).sum().item()

                # Best Student Accuracy:
                best_pred = self.students[self.best_student_idx](data)
                best_pred = best_pred.argmax(dim=1, keepdim=True)
                best_student_correct += best_pred.eq(target.view_as(best_pred)).sum().item()

                # Class Collective Accuracy:
                class_predictions = []
                for student in self.students:
                    pred = student.forward(data)
                    class_predictions.append(pred.argmax(dim=1, keepdim=True))

                class_votes = torch.stack(class_predictions)
                class_final_pred = torch.mode(class_votes, dim=0)[0]
                class_total_correct += class_final_pred.eq(target.view_as(class_final_pred)).sum().item()
                # 8 Students voting on 4 test images:
                # Image 1: [7, 7, 2, 7, 7, 7, 7, 7] → Majority: 7 ✓
                # Image 2: [3, 3, 3, 8, 3, 3, 3, 3] → Majority: 3 ✓
                # Image 3: [1, 9, 1, 1, 9, 1, 1, 1] → Majority: 1 ✓
                # Image 4: [4, 4, 6, 4, 6, 6, 4, 4] → Majority: 4 ✓
                # Result: 4/4 correct through collective wisdom

                total_samples += len(data)

        #  Calculating Accuracies:
        teacher_acc = teacher_correct / total_samples if teacher_correct > 0 else 0
        best_acc = best_student_correct / total_samples
        class_acc = class_total_correct / total_samples

        #  Tracking history & detecting breakthrough:
        self.history['teacher_accuracy'].append(teacher_acc)
        self.history['best_student_accuracy'].append(best_acc)
        self.history['class_avg_accuracy'].append(class_acc)

        if (not self.history['class_surpassed_best'] and
                class_acc > best_acc and
                step > 500):
            self.history['class_surpassed_best'] = True
            self.history['step_when_surpassed'] = step
            print(f"\n BREAKTHROUGH! Class surpassed best student at step {step}!")
            print(f"Class Accuracy: {class_acc:.4f}, Best Student Accuracy: {best_acc:.4f}")

        return teacher_acc, best_acc, class_acc

    def learn_step(self, x, y_true, step):
        # x: Batch of images (e.g., 64 MNIST images)
        # y_true: Correct labels for those images
        # step: Current training step (0 to 5000)

        #  Phase 1: Determine Learning Mode
        if self.teacher.calls_remaining > 0 and step < 2000:
            self.phase = "teacher_phase"
        else:
            self.phase = "peer_phase"

        # Decision Logic:
        # Teacher Phase: Steps 0-2000 AND teacher still has calls remaining
        # Peer Phase: After step 2000 OR when teacher exhausted
        # Setup: Teacher has 1000 calls, so phase transition happens at step 2000

        #  Phase 2: Updating Best Student (Periodically)
        if step % self.eval_interval == 0:
            self.best_student_idx = self.find_best_student()

        # Every 200 steps: Re-evaluating who's the best student
        # Using validation accuracy instead of training loss in order to ensure we're learning from the current top performer that generalizes best

        # Phase 3A: Teacher-Led Learning
        if self.phase == "teacher_phase":
            for i, (student, optimizer) in enumerate(zip(self.students, self.optimizers)):
                optimizer.zero_grad() # resetting gradients: clearing previous update directions
                output = student.forward(x) # forward pass: making predictions on current batch
                loss = F.nll_loss(output, y_true) # calculating loss: comparing with teacher's correct answers
                loss.backward() # calculating how to improve (gradients of loss with respect to weights and biases)
                optimizer.step() # updating weights: actually improving the network
                self.loss_memory[i].append(loss.item()) # recording loss: tracking performance

        # Phase 3B: Peer-to-Peer Learning
        else:
            # Step 1: Best Student Creates "Soft Targets"
            with torch.no_grad(): # Best student makes predictions but doesn't learn (frozen for this step)
                best_student_output = self.students[self.best_student_idx](x) #  Raw scores from the best student
                soft_targets = F.softmax(best_student_output / 2.0, dim=1) # Converting to probabilities with "temperature"
            # Without temperature (normal softmax):
            # [8.0, 2.0, 0.1] → [0.97, 0.03, 0.00] (very confident)
            # With temperature=2.0:
            # [8.0/2, 2.0/2, 0.1/2] = [4.0, 1.0, 0.05] → [0.95, 0.05, 0.00] (softer, more informative)


            # Step 2: Different Learning Paths
            # Best Student's Special Treatment:
            # If teacher still available: Learns from ground truth (stays sharp)
            # If teacher exhausted: Takes a break this step
            # Why: Prevent the expert from being corrupted by teaching others
            for i, (student, optimizer) in enumerate(zip(self.students, self.optimizers)):
                if i == self.best_student_idx:
                    # Best student continues learning from data (if teacher available)
                    if self.teacher.calls_remaining > 0:
                        optimizer.zero_grad()
                        output = student.forward(x)
                        loss = F.nll_loss(output, y_true)
                        loss.backward()
                        optimizer.step()
                        self.loss_memory[i].append(loss.item())
                    continue

                # Step 3: Other Students Learn from Best Student
                optimizer.zero_grad()
                student_output = student.forward(x)

                # Knowledge Distillation Process:
                # Student makes prediction: student_output = student.forward(x)
                # Compare with best student: KL divergence measures how different the thinking is
                # KL Divergence: "How much should I change my thinking to match the expert?"
                distillation_loss = F.kl_div(
                    F.log_softmax(student_output / 2.0, dim=1),
                    soft_targets,
                    reduction='batchmean'
                )

                # Step 4: Combined Learning (If Teacher Available)
                if self.teacher.calls_remaining > 0:
                    classification_loss = F.nll_loss(student_output, y_true)
                    total_loss = 0.7 * distillation_loss + 0.3 * classification_loss
                else:
                    total_loss = distillation_loss

                total_loss.backward()
                optimizer.step()
                self.loss_memory[i].append(total_loss.item())

                # Loss Balancing:
                # 70%: Learning from the best student's thinking style
                # 30%: Learning from ground truth answers (if available)
                # Gradual transition: Pure distillation when teacher leaves