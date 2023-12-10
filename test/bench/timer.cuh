#include <chrono>
#include <string>
#include <iostream>
#include <iomanip>
#include <vector>

namespace bench {

    const size_t PROMPT_LENGTH = 20;

    using Instant = std::chrono::time_point<std::chrono::high_resolution_clock>;
    using Duration = std::chrono::nanoseconds;
    using std::string;
    using std::vector;

    void print_duration(size_t nanoseconds) {
        // right align with "9.3" format
        std::cout << std::right << std::setw(9) << std::setprecision(3) << std::fixed;
        // if less than 1000 ns
        if (nanoseconds < 1000) {
            std::cout << nanoseconds << " ns";
        } else if (nanoseconds < 1000000ull) {
            double us = nanoseconds / 1000.0;
            std::cout << us << " us";
        } else if (nanoseconds < 1000000000ull) {
            double ms = nanoseconds / 1000000.0;
            std::cout << ms << " ms";
        } else {
            double s = nanoseconds / 1000000000.0;
            std::cout << s << " s ";
        }
    }

    void print_duration(const string& prompt, size_t tabs, const Duration& duration, size_t divide) {
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
        size_t count = duration.count();
        if (divide == 1) {
            print_duration(count);
        } else {
            print_duration(count / divide);
            std::cout << " (total ";
            print_duration(count);
            std::cout << ", " << divide << " times)";
        }
        std::cout << std::endl;
    }

    class TimerOnce {
    private:
        Instant start;
        size_t tabs;
    public:
        inline TimerOnce() {
            start = std::chrono::high_resolution_clock::now();
            tabs = 0;
        }
        inline TimerOnce& tab(size_t tabs) {
            this->tabs = tabs;
            return *this;
        }
        inline void finish(const string& prompt) {
            Instant end = std::chrono::high_resolution_clock::now();
            Duration duration = end - start;
            print_duration(prompt, tabs, duration, 1);
        }
    };

    class TimerSingle {
    private:
        Duration accumulated;
        size_t tabs;
        Instant last;
        size_t count;
    public:
        inline TimerSingle() {
            accumulated = Duration(0);
            tabs = 0;
            last = std::chrono::high_resolution_clock::now();
            count = 0;
        }
        inline TimerSingle& tab(size_t tabs) {
            this->tabs = tabs;
            return *this;
        }
        inline void tick() {
            last = std::chrono::high_resolution_clock::now();
        }
        inline void tock() {
            Instant end = std::chrono::high_resolution_clock::now();
            Duration duration = end - last;
            accumulated += duration;
            count++;
        }
        inline void print(const string& prompt) {
            print_duration(prompt, tabs, accumulated, 1);
        }
        inline void print_divided(const string& prompt, size_t divide_override = 0) {
            size_t divide = divide_override == 0 ? count : divide_override;
            print_duration(prompt, tabs, accumulated, divide);
        }
        inline void reset() {
            accumulated = Duration(0);
            count = 0;
            last = std::chrono::high_resolution_clock::now();
        }
    };

    class Timer {
        vector<TimerSingle> timers;
        vector<string> prompts;
        size_t tabs;
    public:
        inline Timer() {
            tabs = 0;
        }
        inline Timer& tab(size_t tabs) {
            this->tabs = tabs;
            for (size_t i = 0; i < timers.size(); i++) {
                timers[i].tab(tabs);
            }
            return *this;
        }
        inline size_t register_timer(const string& prompt) {
            timers.push_back(TimerSingle());
            timers[timers.size() - 1].tab(tabs);
            prompts.push_back(prompt);
            return timers.size() - 1;
        }
        inline void tick(size_t handle = 0) {
            timers[handle].tick();
        }
        inline void tock(size_t handle = 0) {
            timers[handle].tock();
        }
        inline void print() {
            for (size_t i = 0; i < timers.size(); i++) {
                timers[i].print(prompts[i]);
            }
        }
        inline void print_divided(size_t divide_override = 0) {
            for (size_t i = 0; i < timers.size(); i++) {
                timers[i].print_divided(prompts[i], divide_override);
            }
        }
        inline void clear() {
            timers.clear();
            prompts.clear();
        }
        inline void reset() {
            for (size_t i = 0; i < timers.size(); i++) {
                timers[i].reset();
            }
        }
    };

}