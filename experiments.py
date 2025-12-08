from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
import collaborative_classroom as CollaborativeClassroom
import advanced_collaborative_classroom as AdvancedCollaborativeClassroom

class Experiments:

    # Data Pipeline Setup
    # Data Preprocessing Pipeline:
    # ToTensor(): Converts images from (0-255) to (0.0-1.0) tensors
    # Normalize((0.1307,), (0.3081,)): Standardize using MNIST mean & std
    # Why normalize: In order to help neural networks learn faster and more stable
    @staticmethod
    def create_data_stream(batch_size=32, val_batch_size=100):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Dataset Creation with Validation Split:
        # Full training set: 60,000 images
        # Split: 80% training (48,000), 20% validation (12,000)
        # Test set: 10,000 images for final evaluation
        full_train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

        # Using 80% for training, 20% for validation
        train_size = int(0.8 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_train_dataset, [train_size, val_size]
        )

        test_dataset = datasets.MNIST('./data', train=False, transform=transform)

        # Data Loaders - The Collectionless Stream:
        # Key Collectionless Feature:
        # shuffle=True: Random order every epoch = true streaming experience
        # No storage: Data flows through, never accumulated
        # Batch processing: Learn from 64 images, then discard

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

        return train_loader, val_loader, test_loader

    @staticmethod
    def run_experiment():
        print("Initializing Collaborative Classroom Experiment...")

        # Phase 1: Setup
        # 8 students
        # 100-loss window of recent performance
        # Evaluating the best student every 200 steps
        # Batch size 64
        # Validation batch size 100
        #classroom = CollaborativeClassroom.CollaborativeClassroom(
        #    num_students=8,
        #    moving_avg_window=100,
        #    eval_interval_for_best_student=200,
        #    val_batch_size=100
        #)

        classroom = AdvancedCollaborativeClassroom.AdvancedCollaborativeClassroom(
            num_students = 8,
            moving_avg_window = 100,
            eval_interval_for_best_student = 200,
            val_batch_size = 100
        )

        # Getting train, validation, and test loaders
        train_loader, val_loader, test_loader = Experiments.create_data_stream(batch_size=64, val_batch_size=100)

        # Setting up validation batches for consistent evaluation of the best student
        # Virtualization: Storing fixed validation batches in order to ensure fair comparison when selecting the best student
        # Same data used every evaluation = no bias from different samples
        classroom.setup_validation_batches(val_loader, num_batches=5)

        # Phase 2: Training Loop
        # Collectionless Data Streaming:
        # iter(train_loader): Creating an iterator over the data stream
        # next(data_iter): Getting next batch (64 images + labels)
        # StopIteration: When stream ends, restarting from beginning
        # No memory: Each batch processed and forgotten
        steps = 5000
        class_evaluation_every = 500

        print(f"\nStarting training for {steps} steps...")

        # starting training loop on the training dataset (train_loader)
        data_iter = iter(train_loader)
        for step in range(steps):
            try:
                x, y_true = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y_true = next(data_iter)

            # Phase 3: Learning & Evaluation Cycle
            # learning
            classroom.learn_step(x, y_true, step)

            # evaluation of class's students every 500 steps
            if step % class_evaluation_every == 0 or step == steps - 1:
                teacher_acc, best_acc, class_acc = classroom.evaluate_class(test_loader, step)
                print(f"Step {step:4d} | Teacher: {teacher_acc:.3f} | "
                      f"Best Student: {best_acc:.3f} | Class: {class_acc:.3f} | "
                      f"Phase: {classroom.phase} | Best Student ID: {classroom.best_student_idx}")

        # Phase 4: Results Analysis
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

    @staticmethod
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
    classroom = Experiments.run_experiment()
    Experiments.plot_results(classroom)