import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from Models.Teacher import Teacher
from Models.Student import Student


class StabilizedCollaborativeClassroom:
    def __init__(self, num_students=8, moving_avg_window=100,
                 eval_interval_for_best_student=200, val_batch_size=100,
                 corruption_threshold=0.08, reset_threshold=0.15):
        self.num_students = num_students
        self.moving_avg_window = moving_avg_window
        self.eval_interval_for_best_student = eval_interval_for_best_student
        self.val_batch_size = val_batch_size

        # More aggressive corruption detection
        self.corruption_threshold = corruption_threshold  # 8% drop triggers warning
        self.reset_threshold = reset_threshold  # 15% drop triggers reset

        self.students = [Student() for _ in range(num_students)]

        # Enhanced optimizer configuration with different strategies per student
        self.optimizers = []
        for i in range(num_students):
            # Some students use different optimizers for diversity
            if i < num_students // 2:
                optimizer = optim.Adam(self.students[i].parameters(),
                                       lr=0.001, weight_decay=1e-5)
            else:
                optimizer = optim.SGD(self.students[i].parameters(),
                                      lr=0.01, momentum=0.9, weight_decay=1e-4)
            self.optimizers.append(optimizer)

        self.teacher = Teacher()

        # Enhanced memory systems
        self.loss_memory = [deque(maxlen=moving_avg_window) for _ in range(num_students)]
        self.val_acc_history = [[] for _ in range(num_students)]
        self.baseline_accuracies = [0.0] * num_students
        self.peak_accuracies = [0.0] * num_students
        self.recent_performance = [deque(maxlen=15) for _ in range(num_students)]

        # Knowledge stabilization systems
        self.knowledge_snapshots = {}  # Store multiple knowledge snapshots
        self.ensemble_memory = deque(maxlen=20)  # Store ensemble predictions
        self.stability_metrics = deque(maxlen=50)  # Track stability over time

        # Dynamic learning rate scheduling
        self.lr_schedulers = []
        for optimizer in self.optimizers:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.8, patience=3
            )
            self.lr_schedulers.append(scheduler)

        self.best_student_idx = 0
        self.phase = "teacher_phase"

        # Virtualization
        self.validation_batches = []
        self.virtual_validation_pool = []

        # Enhanced tracking
        self.history = {
            'teacher_accuracy': [],
            'best_student_accuracy': [],
            'class_avg_accuracy': [],
            'individual_accuracies': [[] for _ in range(num_students)],
            'ensemble_accuracy': [],  # New: track ensemble performance
            'stability_scores': [],  # New: track stability metrics
            'class_surpassed_best': False,
            'step_when_surpassed': None,
            'corruption_resets': 0,
            'student_resets': [0] * num_students,
            'best_student_changes': 0,
            'knowledge_preservation_events': 0,
        }

    def setup_validation_batches(self, val_loader, num_batches=10):
        self.validation_batches = []
        self.virtual_validation_pool = []

        val_iter = iter(val_loader)

        for _ in range(num_batches):
            try:
                x_val, y_val = next(val_iter)
                self.validation_batches.append((x_val, y_val))
            except StopIteration:
                break

        val_iter = iter(val_loader)
        pool_size = min(25, len(val_loader))
        for _ in range(pool_size):
            try:
                x_val, y_val = next(val_iter)
                self.virtual_validation_pool.append((x_val, y_val))
            except StopIteration:
                break

        print(f"Created {len(self.validation_batches)} validation batches")
        print(f"Created virtual validation pool of {len(self.virtual_validation_pool)} batches")

    def evaluate_student_on_validation(self, student_idx):
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

        accuracy = total_correct / total_samples if total_samples > 0 else 0

        # Enhanced tracking
        self.recent_performance[student_idx].append(accuracy)
        if accuracy > self.peak_accuracies[student_idx]:
            self.peak_accuracies[student_idx] = accuracy

        return accuracy

    def evaluate_student_on_minibatch(self, student_idx, batch_idx=None):
        student = self.students[student_idx]

        if batch_idx is None:
            batch_idx = random.randint(0, len(self.virtual_validation_pool) - 1)

        x_val, y_val = self.virtual_validation_pool[batch_idx]

        with torch.no_grad():
            output = student.forward(x_val)
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(y_val.view_as(pred)).sum().item()
            accuracy = correct / len(x_val)

            # Calculate confidence (entropy-based)
            probs = F.softmax(output, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            confidence = 1.0 - (entropy.mean().item() / np.log(10))  # Normalize by max entropy

        return accuracy, confidence, batch_idx

    def get_moving_average_loss(self, student_idx):
        if len(self.loss_memory[student_idx]) == 0:
            return float('inf')
        return np.mean(list(self.loss_memory[student_idx]))

    def find_best_student(self):
        if not self.validation_batches:
            avg_losses = [self.get_moving_average_loss(i) for i in range(self.num_students)]
            return np.argmin(avg_losses)

        # Multi-criteria evaluation
        val_accuracies = []
        consistency_scores = []
        confidence_scores = []

        for i in range(self.num_students):
            accuracy = self.evaluate_student_on_validation(i)
            val_accuracies.append(accuracy)

            # Consistency (low variance in recent performance)
            if len(self.recent_performance[i]) >= 5:
                recent_accs = list(self.recent_performance[i])[-5:]
                consistency = 1.0 / (1.0 + np.var(recent_accs))
            else:
                consistency = 0.5
            consistency_scores.append(consistency)

            # Confidence
            _, confidence, _ = self.evaluate_student_on_minibatch(i)
            confidence_scores.append(confidence)

        # Convert to numpy arrays and normalize
        val_accuracies = np.array(val_accuracies)
        consistency_scores = np.array(consistency_scores)
        confidence_scores = np.array(confidence_scores)

        # Normalize each metric
        def normalize(arr):
            if arr.max() > arr.min():
                return (arr - arr.min()) / (arr.max() - arr.min())
            return np.ones_like(arr)

        norm_accuracy = normalize(val_accuracies)
        norm_consistency = normalize(consistency_scores)
        norm_confidence = normalize(confidence_scores)

        # Add stability bonus to current best student
        stability_bonus = np.zeros_like(norm_accuracy)
        if hasattr(self, 'best_student_idx'):
            stability_bonus[self.best_student_idx] = 0.1

        # Combined score: 60% accuracy, 20% consistency, 15% confidence, 5% stability
        combined_scores = (0.60 * norm_accuracy +
                           0.20 * norm_consistency +
                           0.15 * norm_confidence +
                           stability_bonus)

        return np.argmax(combined_scores)

    def preserve_knowledge_snapshot(self, step, reason="routine"):
        best_student_state = self.students[self.best_student_idx].state_dict()

        # Store snapshot with metadata
        snapshot_key = f"step_{step}_{reason}"
        self.knowledge_snapshots[snapshot_key] = {
            'state_dict': {name: param.clone() for name, param in best_student_state.items()},
            'step': step,
            'reason': reason,
            'accuracy': self.evaluate_student_on_validation(self.best_student_idx)
        }

        # Keep only the best 5 snapshots
        if len(self.knowledge_snapshots) > 5:
            # Remove the oldest snapshot with lowest accuracy
            worst_key = min(self.knowledge_snapshots.keys(),
                            key=lambda k: self.knowledge_snapshots[k]['accuracy'])
            del self.knowledge_snapshots[worst_key]

        self.history['knowledge_preservation_events'] += 1

    def get_best_knowledge_snapshot(self):
        if not self.knowledge_snapshots:
            return None

        best_key = max(self.knowledge_snapshots.keys(),
                       key=lambda k: self.knowledge_snapshots[k]['accuracy'])
        return self.knowledge_snapshots[best_key]

    def monitor_corruption(self, student_idx, step):
        if len(self.virtual_validation_pool) == 0:
            return False

        if step % 50 == 0:  # More frequent monitoring
            # Evaluate on multiple minibatches
            accuracies = []
            confidences = []
            for _ in range(5):
                acc, conf, _ = self.evaluate_student_on_minibatch(student_idx)
                accuracies.append(acc)
                confidences.append(conf)

            avg_accuracy = np.mean(accuracies)
            avg_confidence = np.mean(confidences)
            std_accuracy = np.std(accuracies)

            # Store history
            self.val_acc_history[student_idx].append(avg_accuracy)
            if len(self.val_acc_history[student_idx]) > 20:
                self.val_acc_history[student_idx].pop(0)

            # Update baseline more aggressively when performance is good
            if avg_accuracy > self.baseline_accuracies[student_idx]:
                self.baseline_accuracies[student_idx] = avg_accuracy

            # Enhanced corruption detection with more indicators
            if len(self.val_acc_history[student_idx]) >= 5:
                recent_avg = np.mean(self.val_acc_history[student_idx][-3:])
                longer_avg = np.mean(self.val_acc_history[student_idx][-5:])
                baseline = max(self.baseline_accuracies[student_idx],
                               self.peak_accuracies[student_idx])

                if baseline > 0.3:
                    corruption_score = 0

                    # 1. Accuracy drop from baseline
                    drop_ratio = (baseline - recent_avg) / baseline
                    if drop_ratio > self.corruption_threshold:
                        corruption_score += 2  # High weight for accuracy drop

                    # 2. Trend analysis - consistently decreasing
                    if len(self.val_acc_history[student_idx]) >= 5:
                        recent_trend = np.polyfit(range(5), self.val_acc_history[student_idx][-5:], 1)[0]
                        if recent_trend < -0.015:  # Negative trend
                            corruption_score += 1

                    # 3. High variance (instability)
                    if std_accuracy > 0.15:
                        corruption_score += 1

                    # 4. Low confidence
                    if avg_confidence < 0.6:
                        corruption_score += 1

                    # 5. Gap with best student
                    if student_idx != self.best_student_idx:
                        best_acc, _, _ = self.evaluate_student_on_minibatch(self.best_student_idx)
                        gap = best_acc - recent_avg
                        if gap > 0.15:
                            corruption_score += 1

                    # 6. Absolute performance threshold
                    if recent_avg < 0.75:  # Below reasonable threshold
                        corruption_score += 1

                    # Reset if corruption score is high enough
                    if corruption_score >= 3:  # Need 3+ indicators
                        print(f"Step {step}: Student {student_idx} corruption detected (score: {corruption_score})")
                        print(f"  Drop: {drop_ratio:.3f}, Acc: {recent_avg:.3f}, Conf: {avg_confidence:.3f}")
                        self.reset_student_with_best_knowledge(student_idx, step)
                        return True

        return False

    def reset_student_with_best_knowledge(self, student_idx, step):
        print(f"Step {step}: Resetting student {student_idx} with best knowledge")

        old_lr = self.optimizers[student_idx].param_groups[0]['lr']

        # Create new student
        self.students[student_idx] = Student()

        # Get best knowledge snapshot
        best_snapshot = self.get_best_knowledge_snapshot()

        if best_snapshot and student_idx != self.best_student_idx:
            new_state = self.students[student_idx].state_dict()
            best_state = best_snapshot['state_dict']

            for name, param in new_state.items():
                if name in best_state:
                    # More aggressive knowledge injection from proven good state
                    new_state[name] = 0.5 * best_state[name] + 0.5 * param

            self.students[student_idx].load_state_dict(new_state)
            print(f"  Injected knowledge from snapshot (acc: {best_snapshot['accuracy']:.3f})")

        # Reset optimizer with appropriate learning rate
        if student_idx < self.num_students // 2:
            self.optimizers[student_idx] = optim.Adam(
                self.students[student_idx].parameters(),
                lr=min(old_lr * 1.5, 0.002), weight_decay=1e-5
            )
        else:
            self.optimizers[student_idx] = optim.SGD(
                self.students[student_idx].parameters(),
                lr=min(old_lr * 1.2, 0.02), momentum=0.9, weight_decay=1e-4
            )

        # Reset scheduler
        self.lr_schedulers[student_idx] = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizers[student_idx], mode='min', factor=0.8, patience=3
        )

        # Clear histories
        self.loss_memory[student_idx].clear()
        self.val_acc_history[student_idx] = []
        self.baseline_accuracies[student_idx] = 0.0
        self.recent_performance[student_idx].clear()

        self.history['corruption_resets'] += 1
        self.history['student_resets'][student_idx] += 1

    def calculate_stability_metric(self, step):
        if len(self.history['class_avg_accuracy']) < 3:
            return 1.0

        recent_accs = self.history['class_avg_accuracy'][-3:]
        stability = 1.0 - np.std(recent_accs)  # Higher stability = lower variance
        self.stability_metrics.append(stability)
        return stability

    def evaluate_class(self, test_loader, step):
        teacher_correct = 0
        best_student_correct = 0
        class_total_correct = 0
        ensemble_correct = 0
        individual_correct = [0] * self.num_students
        total_samples = 0

        with torch.no_grad():
            for data, target in test_loader:
                # Teacher accuracy
                if self.teacher.calls_remaining > 0:
                    teacher_pred = self.students[self.best_student_idx].forward(data)
                    teacher_pred = teacher_pred.argmax(dim=1, keepdim=True)
                    teacher_correct += teacher_pred.eq(target.view_as(teacher_pred)).sum().item()

                # Best Student Accuracy
                best_pred = self.students[self.best_student_idx].forward(data)
                best_pred = best_pred.argmax(dim=1, keepdim=True)
                best_student_correct += best_pred.eq(target.view_as(best_pred)).sum().item()

                # Individual predictions and ensemble
                individual_outputs = []
                class_predictions = []
                for i, student in enumerate(self.students):
                    output = student.forward(data)
                    pred_class = output.argmax(dim=1, keepdim=True)
                    individual_correct[i] += pred_class.eq(target.view_as(pred_class)).sum().item()
                    class_predictions.append(pred_class)
                    individual_outputs.append(F.softmax(output, dim=1))

                # Class majority voting
                class_votes = torch.stack(class_predictions)
                class_final_pred = torch.mode(class_votes, dim=0)[0]
                class_total_correct += class_final_pred.eq(target.view_as(class_final_pred)).sum().item()

                # Enhanced ensemble (weighted average of probabilities)
                ensemble_output = torch.stack(individual_outputs).mean(dim=0)
                ensemble_pred = ensemble_output.argmax(dim=1, keepdim=True)
                ensemble_correct += ensemble_pred.eq(target.view_as(ensemble_pred)).sum().item()

                total_samples += len(data)

        # Calculate accuracies
        teacher_acc = teacher_correct / total_samples if teacher_correct > 0 else 0
        best_acc = best_student_correct / total_samples
        class_acc = class_total_correct / total_samples
        ensemble_acc = ensemble_correct / total_samples
        individual_accs = [correct / total_samples for correct in individual_correct]

        # Calculate stability
        stability = self.calculate_stability_metric(step)

        # Update history
        self.history['teacher_accuracy'].append(teacher_acc)
        self.history['best_student_accuracy'].append(best_acc)
        self.history['class_avg_accuracy'].append(class_acc)
        self.history['ensemble_accuracy'].append(ensemble_acc)
        self.history['stability_scores'].append(stability)

        for i, acc in enumerate(individual_accs):
            self.history['individual_accuracies'][i].append(acc)

        # Preserve knowledge when performing well
        if max(class_acc, ensemble_acc) > 0.92 and step % 500 == 0:
            self.preserve_knowledge_snapshot(step, "high_performance")

        # Detect breakthrough
        best_class_acc = max(class_acc, ensemble_acc)
        if (not self.history['class_surpassed_best'] and
                best_class_acc > best_acc and
                step > 500):
            self.history['class_surpassed_best'] = True
            self.history['step_when_surpassed'] = step
            method = "ensemble" if ensemble_acc > class_acc else "majority"
            print(f"\n🎉 BREAKTHROUGH! Class ({method}) surpassed best student at step {step}!")
            print(f"Class: {best_class_acc:.4f}, Best Student: {best_acc:.4f}")

        return teacher_acc, best_acc, class_acc, individual_accs, ensemble_acc

    def learn_step(self, x, y_true, step):
        # Phase determination
        if self.teacher.calls_remaining > 0 and step < 2000:
            self.phase = "teacher_phase"
        else:
            self.phase = "peer_phase"

        # Corruption monitoring
        if self.phase == "peer_phase":
            for i in range(self.num_students):
                if i != self.best_student_idx:
                    self.monitor_corruption(i, step)

        # Best student selection
        if step % self.eval_interval_for_best_student == 0 and step > 100:
            old_best = self.best_student_idx
            self.best_student_idx = self.find_best_student()
            if old_best != self.best_student_idx:
                self.history['best_student_changes'] += 1
                print(f"Step {step}: New best student is {self.best_student_idx}")
                # Preserve knowledge when best student changes
                if step > 1000:  # Only after some learning
                    self.preserve_knowledge_snapshot(step, "best_student_change")

        teacher_response = self.teacher.teach(x, y_true)

        # Teacher phase
        if self.phase == "teacher_phase":
            if teacher_response is not None:
                for i, (student, optimizer) in enumerate(zip(self.students, self.optimizers)):
                    optimizer.zero_grad()
                    output = student.forward(x)
                    loss = F.nll_loss(output, teacher_response)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                    optimizer.step()
                    self.loss_memory[i].append(loss.item())

        # Enhanced peer phase
        else:
            # Adaptive temperature scheduling based on stability
            stability = self.stability_metrics[-1] if self.stability_metrics else 0.5

            # Base temperature schedule
            max_temp = 2.0
            min_temp = 0.6
            decay_start = 2000
            decay_steps = 3000

            if step < decay_start:
                base_temperature = max_temp
            else:
                progress = min(1.0, (step - decay_start) / decay_steps)
                base_temperature = min_temp + (max_temp - min_temp) * (1 - progress)

            # Adjust temperature based on stability
            temperature = base_temperature * (0.8 + 0.4 * stability)  # More stable = higher temp

            # Create enhanced soft targets
            with torch.no_grad():
                best_student_output = self.students[self.best_student_idx](x)
                soft_targets = F.softmax(best_student_output / temperature, dim=1)

            # Train students
            for i, (student, optimizer, scheduler) in enumerate(
                    zip(self.students, self.optimizers, self.lr_schedulers)):
                if i == self.best_student_idx:
                    # Best student self-improvement
                    if teacher_response is not None:
                        optimizer.zero_grad()
                        output = student.forward(x)
                        loss = F.nll_loss(output, teacher_response)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                        optimizer.step()
                        self.loss_memory[i].append(loss.item())
                        scheduler.step(loss.item())
                    continue

                optimizer.zero_grad()
                student_output = student.forward(x)

                # Knowledge distillation with regularization
                distillation_loss = F.kl_div(
                    F.log_softmax(student_output / temperature, dim=1),
                    soft_targets,
                    reduction='batchmean'
                )

                total_loss = distillation_loss

                # Teacher supervision if available
                if teacher_response is not None:
                    classification_loss = F.nll_loss(student_output, teacher_response)
                    teacher_weight = 0.3 + 0.2 * (1 - stability)  # More teacher weight when unstable
                    total_loss = (1 - teacher_weight) * distillation_loss + teacher_weight * classification_loss

                # Add stability regularization
                l2_reg = 0.0005 * sum(p.pow(2).sum() for p in student.parameters() if p.requires_grad)
                total_loss += l2_reg

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                optimizer.step()
                self.loss_memory[i].append(total_loss.item())
                scheduler.step(total_loss.item())

    def get_detailed_report(self):
        report = "\n" + "=" * 80 + "\n"
        report += "STABILIZED COLLABORATIVE CLASSROOM REPORT\n"
        report += "=" * 80 + "\n"

        report += f"Number of students: {self.num_students}\n"
        report += f"Teacher calls used: {1000 - self.teacher.calls_remaining}/1000\n"
        report += f"Total corruption resets: {self.history['corruption_resets']}\n"
        report += f"Best student changes: {self.history['best_student_changes']}\n"
        report += f"Knowledge preservation events: {self.history['knowledge_preservation_events']}\n"

        # Stability analysis
        if self.stability_metrics:
            avg_stability = np.mean(list(self.stability_metrics))
            final_stability = self.stability_metrics[-1]
            report += f"Average stability: {avg_stability:.4f}\n"
            report += f"Final stability: {final_stability:.4f}\n"

        # Individual student performance
        report += f"\nIndividual Student Performance:\n"
        report += "-" * 50 + "\n"
        for i in range(self.num_students):
            final_acc = self.history['individual_accuracies'][i][-1] if self.history['individual_accuracies'][i] else 0
            resets = self.history['student_resets'][i]
            current_lr = self.optimizers[i].param_groups[0]['lr']
            optimizer_type = type(self.optimizers[i]).__name__
            report += f"Student {i}: Acc: {final_acc:.4f}, Resets: {resets}, LR: {current_lr:.6f} ({optimizer_type})\n"

        # Performance comparison
        if self.history['class_surpassed_best']:
            report += f"\n✅ SUCCESS: Class surpassed best student at step {self.history['step_when_surpassed']}\n"
            final_class_acc = self.history['class_avg_accuracy'][-1]
            final_ensemble_acc = self.history['ensemble_accuracy'][-1] if self.history['ensemble_accuracy'] else 0
            final_best_acc = self.history['best_student_accuracy'][-1]

            best_method = "Ensemble" if final_ensemble_acc > final_class_acc else "Majority Vote"
            best_class_acc = max(final_class_acc, final_ensemble_acc)

            report += f"Final Class Accuracy ({best_method}): {best_class_acc:.4f}\n"
            report += f"Final Best Student Accuracy: {final_best_acc:.4f}\n"
            report += f"Improvement: {best_class_acc - final_best_acc:.4f}\n"
        else:
            report += f"\n❌ Class did not surpass the best student\n"

        # Performance stability in peer phase
        if len(self.history['class_avg_accuracy']) > 4:
            peer_start_idx = 4  # Step 2000
            peer_accs = self.history['class_avg_accuracy'][peer_start_idx:]
            if peer_accs:
                peer_std = np.std(peer_accs)
                peer_decline = peer_accs[0] - peer_accs[-1] if len(peer_accs) > 1 else 0
                report += f"\nPeer Phase Analysis:\n"
                report += f"Performance variance: {peer_std:.4f}\n"
                report += f"Total decline: {peer_decline:.4f}\n"
                report += f"Decline rate: {peer_decline / len(peer_accs):.4f} per evaluation\n"

        report += "=" * 80 + "\n"
        return report