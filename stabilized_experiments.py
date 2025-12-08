from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
from stabilized_collaborative_classroom import StabilizedCollaborativeClassroom


class StabilizedExperiments:

    @staticmethod
    def create_data_stream(batch_size=32, val_batch_size=100):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        full_train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        train_size = int(0.8 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_train_dataset, [train_size, val_size]
        )

        test_dataset = datasets.MNIST('./data', train=False, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

        return train_loader, val_loader, test_loader

    @staticmethod
    def run_stabilized_experiment():
        print("🛡️ Initializing STABILIZED Collaborative Classroom Experiment...")

        # Stabilized classroom with aggressive corruption detection
        classroom = StabilizedCollaborativeClassroom(
            num_students=8,
            moving_avg_window=100,
            eval_interval_for_best_student=200,
            val_batch_size=100,
            corruption_threshold=0.08,  # Very sensitive - 8% drop triggers warning
            reset_threshold=0.15  # 15% drop triggers reset
        )

        train_loader, val_loader, test_loader = StabilizedExperiments.create_data_stream(
            batch_size=64, val_batch_size=100)

        classroom.setup_validation_batches(val_loader, num_batches=10)

        steps = 5000
        class_evaluation_every = 500

        print(f"\n🎯 Starting stabilized training for {steps} steps...")
        print("Advanced stabilization features:")
        print("- Multi-criteria best student selection (accuracy + consistency + confidence)")
        print("- Knowledge snapshots with best-state preservation")
        print("- Enhanced corruption detection (6 indicators)")
        print("- Adaptive temperature based on system stability")
        print("- Mixed optimizer strategies (Adam + SGD)")
        print("- Learning rate scheduling per student")
        print("- Ensemble predictions with weighted averaging")

        data_iter = iter(train_loader)
        for step in range(steps):
            try:
                x, y_true = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y_true = next(data_iter)

            classroom.learn_step(x, y_true, step)

            if step % class_evaluation_every == 0 or step == steps - 1:
                teacher_acc, best_acc, class_acc, individual_accs, ensemble_acc = classroom.evaluate_class(test_loader,
                                                                                                           step)

                resets = classroom.history['corruption_resets']
                changes = classroom.history['best_student_changes']
                snapshots = classroom.history['knowledge_preservation_events']
                stability = classroom.history['stability_scores'][-1] if classroom.history['stability_scores'] else 0

                best_collective = max(class_acc, ensemble_acc)
                method = "Ensemble" if ensemble_acc > class_acc else "Majority"

                print(f"Step {step:4d} | Teacher: {teacher_acc:.3f} | Best: {best_acc:.3f} | "
                      f"Class ({method}): {best_collective:.3f} | Stability: {stability:.3f} | "
                      f"Resets: {resets} | Snapshots: {snapshots} | Phase: {classroom.phase}")

                # Show individual performance and learning rates occasionally
                if step % (class_evaluation_every * 2) == 0 and step > 0:
                    print(f"    Individual: {[f'{acc:.3f}' for acc in individual_accs[:4]]}...")
                    lrs = [opt.param_groups[0]['lr'] for opt in classroom.optimizers[:4]]
                    print(f"    Learning rates: {[f'{lr:.1e}' for lr in lrs]}...")

        report = classroom.get_detailed_report()
        print(report)

        return classroom

    @staticmethod
    def plot_stabilized_results(classroom):
        plt.figure(figsize=(18, 12))

        # Main accuracy plot with ensemble
        plt.subplot(3, 4, 1)
        steps = range(0, len(classroom.history['teacher_accuracy']) * 500, 500)

        plt.plot(steps, classroom.history['teacher_accuracy'], 'g-', label='Teacher', alpha=0.7)
        plt.plot(steps, classroom.history['best_student_accuracy'], 'r-', label='Best Student', linewidth=2)
        plt.plot(steps, classroom.history['class_avg_accuracy'], 'b-', label='Majority Vote', linewidth=2)

        if classroom.history['ensemble_accuracy']:
            plt.plot(steps, classroom.history['ensemble_accuracy'], 'm-', label='Ensemble', linewidth=2)

        if classroom.history['class_surpassed_best']:
            surpass_step = classroom.history['step_when_surpassed']
            plt.axvline(x=surpass_step, color='purple', linestyle='--',
                        label=f'Breakthrough (step {surpass_step})')

        plt.xlabel('Training Step')
        plt.ylabel('Accuracy')
        plt.title('Stabilized Learning Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0.8, 1.0)  # Focus on high-accuracy range


    @staticmethod
    def analyze_degradation(classroom):
        print("\n" + "=" * 70)
        print("DEGRADATION ANALYSIS")
        print("=" * 70)

        if len(classroom.history['class_avg_accuracy']) <= 4:
            print("Not enough data for peer phase analysis")
            return

        # Extract peer phase data
        peer_start = 4  # Step 2000
        peer_class = classroom.history['class_avg_accuracy'][peer_start:]
        peer_best = classroom.history['best_student_accuracy'][peer_start:]

        if len(peer_class) < 2:
            print("Not enough peer phase data")
            return

        # Calculate degradation metrics
        class_initial = peer_class[0]
        class_final = peer_class[-1]
        class_degradation = class_initial - class_final

        best_initial = peer_best[0]
        best_final = peer_best[-1]
        best_degradation = best_initial - best_final

        # Calculate degradation rate (per 1000 steps)
        peer_steps = len(peer_class) - 1
        steps_per_eval = 500
        total_peer_steps = peer_steps * steps_per_eval

        class_rate = (class_degradation / total_peer_steps) * 1000 if total_peer_steps > 0 else 0
        best_rate = (best_degradation / total_peer_steps) * 1000 if total_peer_steps > 0 else 0

        print(f"Peer Phase Performance Analysis:")
        print(f"Class Performance:")
        print(f"  Initial: {class_initial:.4f}")
        print(f"  Final: {class_final:.4f}")
        print(f"  Total decline: {class_degradation:.4f}")
        print(f"  Decline rate: {class_rate:.4f} per 1000 steps")

        print(f"\nBest Student Performance:")
        print(f"  Initial: {best_initial:.4f}")
        print(f"  Final: {best_final:.4f}")
        print(f"  Total decline: {best_degradation:.4f}")
        print(f"  Decline rate: {best_rate:.4f} per 1000 steps")

        # Stability analysis
        class_variance = np.var(peer_class)
        best_variance = np.var(peer_best)

        print(f"\nStability Metrics:")
        print(f"  Class variance: {class_variance:.6f}")
        print(f"  Best student variance: {best_variance:.6f}")

        # Overall assessment
        print(f"\nOverall Assessment:")
        if class_degradation < 0.01 and best_degradation < 0.01:
            print("✅ EXCELLENT: Minimal degradation in peer phase")
        elif class_degradation < 0.02 and best_degradation < 0.02:
            print("✅ GOOD: Acceptable degradation levels")
        elif class_degradation < 0.05:
            print("⚠️  MODERATE: Some degradation present but controlled")
        else:
            print("❌ POOR: Significant degradation still occurring")

        # Recommendations
        if class_degradation > 0.02:
            print("\nRecommendations for further improvement:")
            print("- Increase knowledge preservation frequency")
            print("- Lower corruption detection thresholds")
            print("- Add more ensemble diversity")
            print("- Implement stronger regularization")

        print("=" * 70)


if __name__ == "__main__":
    print("Running STABILIZED Collaborative Classroom experiment...")
    stabilized_classroom = StabilizedExperiments.run_stabilized_experiment()

    StabilizedExperiments.plot_stabilized_results(stabilized_classroom)
    StabilizedExperiments.analyze_degradation(stabilized_classroom)

    print("\n🛡️ Stabilized experiment completed!")