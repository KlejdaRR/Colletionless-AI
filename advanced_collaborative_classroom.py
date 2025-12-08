import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from Models.Teacher import Teacher
from Models.Student import Student

class AdvancedCollaborativeClassroom:
    def __init__(self, num_students=8, moving_avg_window=100,
                 eval_interval_for_best_student=200, val_batch_size=100,
                 corruption_threshold=0.15, reset_threshold=0.05):
        self.num_students = num_students
        self.moving_avg_window = moving_avg_window
        self.eval_interval_for_best_student = eval_interval_for_best_student
        self.val_batch_size = val_batch_size

        # New: Corruption detection thresholds
        self.corruption_threshold = corruption_threshold  # Accuracy drop to trigger warning
        self.reset_threshold = reset_threshold  # Accuracy drop to trigger reset

        self.students = [Student() for _ in range(num_students)]
        self.optimizers = [optim.Adam(student.parameters(), lr=0.001) for student in self.students]
        self.teacher = Teacher()

        # Loss memory for each student
        self.loss_memory = [deque(maxlen=moving_avg_window) for _ in range(num_students)]

        # NEW: Validation accuracy history for corruption detection
        self.val_acc_history = [[] for _ in range(num_students)]
        self.baseline_accuracies = [0.0] * num_students  # Store peak performance

        self.best_student_idx = 0
        self.phase = "teacher_phase"

        # Virtualization: Stored validation data
        self.validation_batches = []
        self.virtual_validation_pool = []  # NEW: Pool of validation samples for monitoring

        # Tracking
        self.history = {
            'teacher_accuracy': [],
            'best_student_accuracy': [],
            'class_avg_accuracy': [],
            'class_surpassed_best': False,
            'step_when_surpassed': None,
            'corruption_resets': 0,  # NEW: Track resets
            'student_resets': [0] * num_students  # NEW: Track per-student resets
        }

    def setup_validation_batches(self, val_loader, num_batches=5):
        """Setup validation batches for evaluation AND virtualization"""
        self.validation_batches = []
        self.virtual_validation_pool = []  # NEW: For corruption monitoring

        val_iter = iter(val_loader)

        # Store regular validation batches
        for _ in range(num_batches):
            try:
                x_val, y_val = next(val_iter)
                self.validation_batches.append((x_val, y_val))
            except StopIteration:
                break

        # NEW: Create a larger pool of validation samples for virtualization
        # We'll use small minibatches of 100 with permutations
        val_iter = iter(val_loader)
        pool_size = min(20, len(val_loader))  # Pool of 20 batches for diversity
        for _ in range(pool_size):
            try:
                x_val, y_val = next(val_iter)
                self.virtual_validation_pool.append((x_val, y_val))
            except StopIteration:
                break

        print(f"Created {len(self.validation_batches)} validation batches for best student selection")
        print(f"Created virtual validation pool of {len(self.virtual_validation_pool)} batches for corruption monitoring")

    # ADD THIS MISSING METHOD:
    def evaluate_student_on_validation(self, student_idx):
        """Evaluate a single student on all validation batches"""
        student = self.students[student_idx]
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for x_val, y_val in self.validation_batches:
                output = student.forward(x_val)
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(y_val.view_as(pred)).sum().item()
                total_correct += correct
                total_samples += len(x_val)

        return total_correct / total_samples if total_samples > 0 else 0

    def evaluate_student_on_minibatch(self, student_idx, batch_idx=None):
        """Evaluate a student on a small minibatch from virtual pool"""
        student = self.students[student_idx]

        # NEW: Use random permutation of batches
        if batch_idx is None:
            batch_idx = random.randint(0, len(self.virtual_validation_pool) - 1)

        x_val, y_val = self.virtual_validation_pool[batch_idx]

        with torch.no_grad():
            output = student.forward(x_val)
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(y_val.view_as(pred)).sum().item()
            accuracy = correct / len(x_val)

        return accuracy, batch_idx

    # ADD THIS MISSING METHOD TOO:
    def get_moving_average_loss(self, student_idx):
        """Get moving average loss for a student"""
        if len(self.loss_memory[student_idx]) == 0:
            return float('inf')
        return np.mean(list(self.loss_memory[student_idx]))

    def find_best_student(self):
        """Find the best student based on validation performance"""
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

    def monitor_corruption(self, student_idx, step):
        """Monitor and handle student corruption using virtualization"""
        if len(self.virtual_validation_pool) == 0:
            return False  # No virtualization data available

        # NEW: Periodically check student health using small minibatches
        if step % 100 == 0:  # Check every 100 steps
            # Evaluate on 3 random minibatches for robustness
            accuracies = []
            for _ in range(3):
                acc, _ = self.evaluate_student_on_minibatch(student_idx)
                accuracies.append(acc)

            avg_accuracy = np.mean(accuracies)

            # Store accuracy history
            self.val_acc_history[student_idx].append(avg_accuracy)
            if len(self.val_acc_history[student_idx]) > 10:
                self.val_acc_history[student_idx].pop(0)

            # Update baseline if this is higher
            if avg_accuracy > self.baseline_accuracies[student_idx]:
                self.baseline_accuracies[student_idx] = avg_accuracy

            # Check for significant drop in accuracy
            if len(self.val_acc_history[student_idx]) >= 5:
                recent_avg = np.mean(self.val_acc_history[student_idx][-5:])
                baseline = self.baseline_accuracies[student_idx]

                if baseline > 0:  # Only check if we have a baseline
                    drop_ratio = (baseline - recent_avg) / baseline

                    # Corruption detected!
                    if drop_ratio > self.corruption_threshold:
                        print(f"Step {step}: Student {student_idx} showing corruption signs (drop: {drop_ratio:.3f})")

                        # Apply corrective measures
                        if drop_ratio > self.reset_threshold:
                            self.reset_student(student_idx, step)
                            return True

        return False

    def reset_student(self, student_idx, step):
        """Reset a corrupted student while preserving some knowledge"""
        print(f"Step {step}: Resetting corrupted student {student_idx}")

        # NEW: Soft reset - reinitialize but keep optimizer state
        # Store current state for comparison
        old_state = self.students[student_idx].state_dict().copy()

        # Create new student with same architecture
        self.students[student_idx] = Student()

        # NEW: Knowledge injection from best student
        if student_idx != self.best_student_idx:
            # Transfer some knowledge from best student
            best_state = self.students[self.best_student_idx].state_dict()
            new_state = self.students[student_idx].state_dict()

            # Blend: 30% from best student, 70% fresh initialization
            for key in new_state.keys():
                if 'weight' in key or 'bias' in key:
                    new_state[key] = 0.3 * best_state[key] + 0.7 * new_state[key]

            self.students[student_idx].load_state_dict(new_state)

        # Reset optimizer
        self.optimizers[student_idx] = optim.Adam(
            self.students[student_idx].parameters(),
            lr=0.001
        )

        # Clear history
        self.loss_memory[student_idx].clear()
        self.val_acc_history[student_idx] = []
        self.baseline_accuracies[student_idx] = 0.0

        # Track reset
        self.history['corruption_resets'] += 1
        self.history['student_resets'][student_idx] += 1

        return True

    # ADD THIS MISSING METHOD (if needed for evaluation):
    def evaluate_class(self, test_loader, step):
        """Evaluate teacher, best student, and class collective accuracy"""
        teacher_correct = 0
        best_student_correct = 0
        class_total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in test_loader:
                # Teacher accuracy (if teacher still has calls)
                if self.teacher.calls_remaining > 0:
                    teacher_pred = self.students[self.best_student_idx].forward(data)
                    teacher_pred = teacher_pred.argmax(dim=1, keepdim=True)
                    teacher_correct += teacher_pred.eq(target.view_as(teacher_pred)).sum().item()

                # Best Student Accuracy
                best_pred = self.students[self.best_student_idx].forward(data)
                best_pred = best_pred.argmax(dim=1, keepdim=True)
                best_student_correct += best_pred.eq(target.view_as(best_pred)).sum().item()

                # Class Collective Accuracy
                class_predictions = []
                for student in self.students:
                    pred = student.forward(data)
                    class_predictions.append(pred.argmax(dim=1, keepdim=True))

                class_votes = torch.stack(class_predictions)
                class_final_pred = torch.mode(class_votes, dim=0)[0]
                class_total_correct += class_final_pred.eq(target.view_as(class_final_pred)).sum().item()

                total_samples += len(data)

        # Calculate Accuracies
        teacher_acc = teacher_correct / total_samples if teacher_correct > 0 else 0
        best_acc = best_student_correct / total_samples
        class_acc = class_total_correct / total_samples

        # Tracking history & detecting breakthrough
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
        """Modified learn_step with corruption monitoring"""
        # Phase 1: Determine Learning Mode
        if self.teacher.calls_remaining > 0 and step < 2000:
            self.phase = "teacher_phase"
        else:
            self.phase = "peer_phase"

        # NEW: Monitor for corruption in peer phase
        if self.phase == "peer_phase":
            for i in range(self.num_students):
                if i != self.best_student_idx:  # Don't monitor best student
                    self.monitor_corruption(i, step)

        # Phase 2: Updating Best Student (Periodically)
        if step % self.eval_interval_for_best_student == 0:
            old_best = self.best_student_idx
            self.best_student_idx = self.find_best_student()
            if old_best != self.best_student_idx:
                print(f"Step {step}: New best student is {self.best_student_idx}")

        # Get teacher's response
        teacher_response = self.teacher.teach(x, y_true)

        # Phase 3A: Teacher-Led Learning
        if self.phase == "teacher_phase":
            if teacher_response is not None:
                for i, (student, optimizer) in enumerate(zip(self.students, self.optimizers)):
                    optimizer.zero_grad()
                    output = student.forward(x)
                    loss = F.nll_loss(output, teacher_response)
                    loss.backward()
                    optimizer.step()
                    self.loss_memory[i].append(loss.item())
            else:
                self.phase = "peer_phase"

        # Phase 3B: Peer-to-Peer Learning
        if self.phase == "peer_phase":
            # Step 1: Best Student Creates "Soft Targets"
            with torch.no_grad():
                best_student_output = self.students[self.best_student_idx](x)
                # NEW: Add temperature scheduling
                temperature = max(0.5, 2.0 * (1 - step / 5000))  # Gradually reduce temperature
                soft_targets = F.softmax(best_student_output / temperature, dim=1)

            # Step 2: Different Learning Paths
            for i, (student, optimizer) in enumerate(zip(self.students, self.optimizers)):
                if i == self.best_student_idx:
                    # Best student learns from teacher if available
                    if teacher_response is not None:
                        optimizer.zero_grad()
                        output = student.forward(x)
                        loss = F.nll_loss(output, teacher_response)
                        loss.backward()
                        optimizer.step()
                        self.loss_memory[i].append(loss.item())
                    continue

                # Step 3: Other Students Learn from Best Student
                optimizer.zero_grad()
                student_output = student.forward(x)

                # Knowledge Distillation with adaptive weighting
                distillation_loss = F.kl_div(
                    F.log_softmax(student_output / temperature, dim=1),
                    soft_targets,
                    reduction='batchmean'
                )

                # NEW: Adaptive loss weighting based on student confidence
                if teacher_response is not None:
                    classification_loss = F.nll_loss(student_output, teacher_response)

                    # Calculate student confidence
                    with torch.no_grad():
                        student_probs = F.softmax(student_output, dim=1)
                        confidence = student_probs.max(dim=1)[0].mean().item()

                    # More weight to teacher when student is less confident
                    teacher_weight = max(0.3, 1.0 - confidence)
                    dist_weight = 1.0 - teacher_weight

                    total_loss = dist_weight * distillation_loss + teacher_weight * classification_loss
                else:
                    total_loss = distillation_loss

                total_loss.backward()
                optimizer.step()
                self.loss_memory[i].append(total_loss.item())