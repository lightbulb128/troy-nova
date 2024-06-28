#include "timer.h"

namespace troy::bench {

    void print_duration_max_mean(const string& prompt, size_t tabs, size_t count, const Duration& max, const Duration& mean, size_t divide) {
// print tabs * 2 spaces
        for (size_t i = 0; i < tabs; i++) {
            std::cout << "  ";
        }
        // print prompt
        std::cout << prompt;
        // print spaces to fill PROMPT_LENGTH
        size_t current_used = prompt.length() + tabs * 2;
        if (current_used < PROMPT_LENGTH) {
            for (size_t i = 0; i < PROMPT_LENGTH - current_used; i++) {
                std::cout << " ";
            }
        }
        // print colon
        std::cout << ": ";
        // print duration
        size_t max_count = max.count();
        size_t mean_count = mean.count();
        if (divide == 1) {
            std::cout << "max ";
            print_duration(max_count);
            std::cout << " / thread, avg ";
            print_duration(mean_count);
            std::cout << " / thread ";
        } else {
            std::cout << "max ";
            print_duration(max_count / divide);
            std::cout << " / op, avg ";
            print_duration(mean_count / divide);
            std::cout << " / op (total max ";
            print_duration(max_count);
            std::cout << " / thread, avg ";
            print_duration(mean_count);
            std::cout << " / thread, " << divide << " times)";
        }
        std::cout << std::endl;
    }

    TimerThreaded::TimerThreaded(const vector<Timer>& timers) {
        prompts.clear();
        maxs.clear();
        averages.clear();
        vector<size_t> occurences;
        vector<Duration> sums;

        tabs = 0;

        for (const Timer& timer : timers) {
            if (timer.tabs > tabs) {
                tabs = timer.tabs;
            }
            for (size_t i = 0; i < timer.prompts.size(); i++) {
                int index = -1;
                for (size_t j = 0; j < prompts.size(); j++) {
                    if (prompts[j] == timer.prompts[i]) {
                        index = static_cast<int>(j);
                        break;
                    }
                }
                if (index == -1) {
                    prompts.push_back(timer.prompts[i]);
                    maxs.push_back(timer.timers[i].accumulated);
                    sums.push_back(Duration::zero());
                    occurences.push_back(0);
                    index = prompts.size() - 1;
                }
                sums[index] += timer.timers[i].accumulated;
                occurences[index]++;
                if (timer.timers[i].accumulated > maxs[index]) {
                    maxs[index] = timer.timers[i].accumulated;
                }
            }
        }

        for (size_t i = 0; i < prompts.size(); i++) {
            averages.push_back(sums[i] / occurences[i]);
        }
    }

    void TimerThreaded::print() const {
        for (size_t i = 0; i < prompts.size(); i++) {
            print_duration_max_mean(prompts[i], tabs, 1, maxs[i], averages[i], 1);
        }
    }

    void TimerThreaded::print_divided(size_t divide) const {
        for (size_t i = 0; i < prompts.size(); i++) {
            print_duration_max_mean(prompts[i], tabs, 1, maxs[i], averages[i], divide);
        }
    }

}