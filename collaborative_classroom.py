
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from Models.Teacher import Teacher
from Models.Student import Student

class CollaborativeClassroom:
    def __init__(self, num_students=10, moving_avg_window=50, eval_interval=100):
        self.num_students = num_students
        self.moving_avg_window = moving_avg_window
        self.eval_interval = eval_interval

        self.students = [Student() for _ in range(num_students)]
        self.optimizers = [optim.Adam(student.parameters(), lr=0.001) for student in self.students]

        self.teacher = Teacher()

        self.loss_memory = [deque(maxlen=moving_avg_window) for _ in range(num_students)]
        self.best_student_idx = 0
        self.phase = "teacher_phase"

        self.history = {
            'teacher_accuracy': [],
            'best_student_accuracy': [],
            'class_avg_accuracy': [],
            'class_surpassed_best': False,
            'step_when_surpassed': None
        }

    def get_moving_average_loss(self, student_idx):
        if len(self.loss_memory[student_idx]) == 0:
            return float('inf')
        return np.mean(list(self.loss_memory[student_idx]))

    def find_best_student(self):
        avg_losses = [self.get_moving_average_loss(i) for i in range(self.num_students)]
        return np.argmin(avg_losses)

    def evaluate_students(self, test_loader, step):
        teacher_correct = 0
        best_student_correct = 0
        class_total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in test_loader:
                if self.teacher.calls_remaining > 0:
                    teacher_pred = self.students[self.best_student_idx](data)
                    teacher_pred = teacher_pred.argmax(dim=1, keepdim=True)
                    teacher_correct += teacher_pred.eq(target.view_as(teacher_pred)).sum().item()

                best_pred = self.students[self.best_student_idx](data)
                best_pred = best_pred.argmax(dim=1, keepdim=True)
                best_student_correct += best_pred.eq(target.view_as(best_pred)).sum().item()

                class_predictions = []
                for student in self.students:
                    pred = student(data)
                    class_predictions.append(pred.argmax(dim=1, keepdim=True))

                class_votes = torch.stack(class_predictions)
                class_final_pred = torch.mode(class_votes, dim=0)[0]
                class_total_correct += class_final_pred.eq(target.view_as(class_final_pred)).sum().item()

                total_samples += len(data)

        teacher_acc = teacher_correct / total_samples if teacher_correct > 0 else 0
        best_acc = best_student_correct / total_samples
        class_acc = class_total_correct / total_samples

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

        if self.teacher.calls_remaining > 0 and step < 2000:
            self.phase = "teacher_phase"
        else:
            self.phase = "peer_phase"

        if step % self.eval_interval == 0:
            self.best_student_idx = self.find_best_student()

        if self.phase == "teacher_phase":
            for i, (student, optimizer) in enumerate(zip(self.students, self.optimizers)):
                optimizer.zero_grad()
                output = student(x)
                loss = F.nll_loss(output, y_true)
                loss.backward()
                optimizer.step()
                self.loss_memory[i].append(loss.item())

        else:
            with torch.no_grad():
                best_student_output = self.students[self.best_student_idx](x)
                soft_targets = F.softmax(best_student_output / 2.0, dim=1)

            for i, (student, optimizer) in enumerate(zip(self.students, self.optimizers)):
                if i == self.best_student_idx:
                    # Best student continues learning from data (if teacher available)
                    if self.teacher.calls_remaining > 0:
                        optimizer.zero_grad()
                        output = student(x)
                        loss = F.nll_loss(output, y_true)
                        loss.backward()
                        optimizer.step()
                        self.loss_memory[i].append(loss.item())
                    continue

                optimizer.zero_grad()
                student_output = student(x)

                distillation_loss = F.kl_div(
                    F.log_softmax(student_output / 2.0, dim=1),
                    soft_targets,
                    reduction='batchmean'
                )

                if self.teacher.calls_remaining > 0:
                    classification_loss = F.nll_loss(student_output, y_true)
                    total_loss = 0.7 * distillation_loss + 0.3 * classification_loss
                else:
                    total_loss = distillation_loss

                total_loss.backward()
                optimizer.step()
                self.loss_memory[i].append(total_loss.item())