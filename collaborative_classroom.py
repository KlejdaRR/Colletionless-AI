import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


class Student(nn.Module):

    def __init__(self):
        super(Student, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the image
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class Teacher:

    def __init__(self):
        self.calls_remaining = 1000

    def teach(self, x, y_true):
        if self.calls_remaining > 0:
            self.calls_remaining -= 1
            return y_true
        return None


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

        if not self.history['class_surpassed_best'] and class_acc > best_acc:
            self.history['class_surpassed_best'] = True
            self.history['step_when_surpassed'] = step
            print(f"\n🎉 BREAKTHROUGH! Class surpassed best student at step {step}!")
            print(f"Class Accuracy: {class_acc:.4f}, Best Student Accuracy: {best_acc:.4f}")

        return teacher_acc, best_acc, class_acc

    def learn_step(self, x, y_true, step):

        if self.teacher.calls_remaining > 0 and step < 2000:  # Extended teacher phase
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
                soft_targets = F.softmax(best_student_output / 2.0, dim=1)  # Temperature = 2.0

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


def create_data_stream(batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader, test_loader


def run_experiment():
    print("Initializing Collaborative Classroom Experiment...")
    print("Collectionless AI Principles:")
    print("No data storage between steps")
    print("Online learning from stream")
    print("Peer-to-peer knowledge transfer")
    print("Evolving internal representations")

    classroom = CollaborativeClassroom(num_students=8, moving_avg_window=100, eval_interval=200)
    train_loader, test_loader = create_data_stream(batch_size=64)

    steps = 5000
    eval_every = 500

    print(f"\nStarting training for {steps} steps...")

    data_iter = iter(train_loader)
    for step in range(steps):
        try:
            x, y_true = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y_true = next(data_iter)

        classroom.learn_step(x, y_true, step)

        if step % eval_every == 0 or step == steps - 1:
            teacher_acc, best_acc, class_acc = classroom.evaluate_students(test_loader, step)
            print(f"Step {step:4d} | Teacher: {teacher_acc:.3f} | "
                  f"Best Student: {best_acc:.3f} | Class: {class_acc:.3f} | "
                  f"Phase: {classroom.phase}")

    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"Number of students: {classroom.num_students}")
    print(f"Teacher calls used: {1000 - classroom.teacher.calls_remaining}/1000")

    if classroom.history['class_surpassed_best']:
        print(f"SUCCESS: Class surpassed best student at step {classroom.history['step_when_surpassed']}")
        final_class_acc = classroom.history['class_avg_accuracy'][-1]
        final_best_acc = classroom.history['best_student_accuracy'][-1]
        print(f"Final Class Accuracy: {final_class_acc:.4f}")
        print(f"Final Best Student Accuracy: {final_best_acc:.4f}")
        print(f"Improvement: {final_class_acc - final_best_acc:.4f}")
    else:
        print("Class did not surpass the best student in this run")

    return classroom


def plot_results(classroom):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    steps = range(0, len(classroom.history['teacher_accuracy']) * 500, 500)
    plt.plot(steps, classroom.history['teacher_accuracy'], 'g-', label='Teacher', alpha=0.7)
    plt.plot(steps, classroom.history['best_student_accuracy'], 'r-', label='Best Student', linewidth=2)
    plt.plot(steps, classroom.history['class_avg_accuracy'], 'b-', label='Class Average', linewidth=2)

    if classroom.history['class_surpassed_best']:
        surpass_step = classroom.history['step_when_surpassed']
        plt.axvline(x=surpass_step, color='purple', linestyle='--',
                    label=f'Class Surpassed Best (step {surpass_step})')

    plt.xlabel('Training Step')
    plt.ylabel('Accuracy')
    plt.title('Collaborative Classroom Learning Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    for i in range(min(4, classroom.num_students)):
        losses = list(classroom.loss_memory[i])
        plt.plot(range(len(losses)), losses, alpha=0.6, label=f'Student {i + 1}')
    plt.xlabel('Recent Steps')
    plt.ylabel('Loss')
    plt.title('Student Losses (Moving Window)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    teacher_steps = 2000
    phases = ['Teacher Phase'] * (teacher_steps // 500) + ['Peer Phase'] * (len(steps) - teacher_steps // 500)
    plt.plot(steps, [0.5] * len(steps), 'k-', linewidth=3)
    plt.fill_between(steps[:len(phases)], 0, 1, where=[p == 'Teacher Phase' for p in phases],
                     alpha=0.3, color='green', label='Teacher Phase')
    plt.fill_between(steps[:len(phases)], 0, 1, where=[p == 'Peer Phase' for p in phases],
                     alpha=0.3, color='orange', label='Peer Phase')
    plt.xlabel('Training Step')
    plt.title('Learning Phases')
    plt.yticks([])
    plt.legend()

    plt.subplot(2, 2, 4)
    categories = ['Best Student', 'Class Collective']
    final_accuracies = [classroom.history['best_student_accuracy'][-1],
                        classroom.history['class_avg_accuracy'][-1]]
    colors = ['red', 'blue']
    bars = plt.bar(categories, final_accuracies, color=colors, alpha=0.7)
    plt.ylabel('Final Accuracy')
    plt.title('Final Performance Comparison')

    for bar, acc in zip(bars, final_accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{acc:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('collaborative_classroom_results.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    classroom = run_experiment()
    plot_results(classroom)

    print("\nExperiment completed! Check 'collaborative_classroom_results.png' for the plots.")