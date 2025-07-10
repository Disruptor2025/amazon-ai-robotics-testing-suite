#!/bin/bash

# Amazon AI & Robotics Testing Suite - System Monitor
# This script monitors system resources, services, and application health

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
CONFIG_DIR="$PROJECT_ROOT/config"
LOGS_DIR="$PROJECT_ROOT/logs"
MONITORING_DIR="$PROJECT_ROOT/monitoring"

# Load configuration
if [ -f "$CONFIG_DIR/config.env" ]; then
    source "$CONFIG_DIR/config.env"
fi

# Default values
MONITORING_INTERVAL=${MONITORING_INTERVAL:-60}
ALERT_EMAIL=${ALERT_EMAIL:-"admin@example.com"}
LOG_FILE="$LOGS_DIR/system_monitor.log"
METRICS_FILE="$MONITORING_DIR/metrics.json"

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$LOG_FILE"
}

# Function to get system metrics
get_system_metrics() {
    local timestamp=$(date +%s)
    local metrics="{}"
    
    # CPU usage
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    
    # Memory usage
    local mem_info=$(free -m | grep Mem)
    local mem_total=$(echo $mem_info | awk '{print $2}')
    local mem_used=$(echo $mem_info | awk '{print $3}')
    local mem_usage=$(echo "scale=2; $mem_used * 100 / $mem_total" | bc)
    
    # Disk usage
    local disk_usage=$(df / | tail -1 | awk '{print $5}' | cut -d'%' -f1)
    
    # Load average
    local load_avg=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | tr -d ',')
    
    # Network usage
    local net_rx=$(cat /proc/net/dev | grep eth0 | awk '{print $2}')
    local net_tx=$(cat /proc/net/dev | grep eth0 | awk '{print $10}')
    
    # Create JSON metrics
    metrics=$(cat << EOF
{
    "timestamp": $timestamp,
    "cpu_usage": $cpu_usage,
    "memory_usage": $mem_usage,
    "disk_usage": $disk_usage,
    "load_average": $load_avg,
    "network_rx": $net_rx,
    "network_tx": $net_tx,
    "memory_total_mb": $mem_total,
    "memory_used_mb": $mem_used
}
EOF
)
    
    echo "$metrics"
}

# Function to check service health
check_service_health() {
    local service_name="$1"
    local status="unknown"
    
    case "$service_name" in
        "docker")
            if systemctl is-active --quiet docker; then
                status="healthy"
            else
                status="unhealthy"
            fi
            ;;
        "postgres")
            if docker ps | grep -q postgres; then
                status="healthy"
            else
                status="unhealthy"
            fi
            ;;
        "redis")
            if docker ps | grep -q redis; then
                status="healthy"
            else
                status="unhealthy"
            fi
            ;;
        "prometheus")
            if curl -s http://localhost:9090/-/healthy >/dev/null 2>&1; then
                status="healthy"
            else
                status="unhealthy"
            fi
            ;;
        "grafana")
            if curl -s http://localhost:3000/api/health >/dev/null 2>&1; then
                status="healthy"
            else
                status="unhealthy"
            fi
            ;;
        *)
            status="unknown"
            ;;
    esac
    
    echo "$status"
}

# Function to check AWS services
check_aws_services() {
    local aws_status="{}"
    
    # Check if AWS CLI is configured
    if ! command -v aws >/dev/null 2>&1; then
        echo '{"aws_cli": "not_installed"}'
        return
    fi
    
    # Check SageMaker endpoints
    local sagemaker_endpoints="[]"
    if aws sagemaker list-endpoints --query 'Endpoints[].EndpointName' --output text >/dev/null 2>&1; then
        sagemaker_endpoints=$(aws sagemaker list-endpoints --query 'Endpoints[].{Name:EndpointName,Status:EndpointStatus}' --output json 2>/dev/null || echo "[]")
    fi
    
    # Check S3 buckets
    local s3_buckets="[]"
    if aws s3 ls >/dev/null 2>&1; then
        s3_buckets=$(aws s3 ls --output json 2>/dev/null || echo "[]")
    fi
    
    aws_status=$(cat << EOF
{
    "sagemaker_endpoints": $sagemaker_endpoints,
    "s3_buckets": $s3_buckets
}
EOF
)
    
    echo "$aws_status"
}

# Function to check application health
check_application_health() {
    local app_status="{}"
    
    # Check if Python virtual environment exists
    local venv_status="unhealthy"
    if [ -d "$PROJECT_ROOT/venv" ]; then
        venv_status="healthy"
    fi
    
    # Check if test results directory exists
    local test_results_status="unhealthy"
    if [ -d "$PROJECT_ROOT/tests/results" ]; then
        test_results_status="healthy"
    fi
    
    # Check recent log files
    local log_status="unhealthy"
    if [ -f "$LOG_FILE" ] && [ $(find "$LOG_FILE" -mmin -60 | wc -l) -gt 0 ]; then
        log_status="healthy"
    fi
    
    app_status=$(cat << EOF
{
    "virtual_environment": "$venv_status",
    "test_results": "$test_results_status",
    "logging": "$log_status"
}
EOF
)
    
    echo "$app_status"
}

# Function to generate alerts
generate_alerts() {
    local metrics="$1"
    local alerts="[]"
    
    # Parse metrics
    local cpu_usage=$(echo "$metrics" | jq -r '.cpu_usage')
    local memory_usage=$(echo "$metrics" | jq -r '.memory_usage')
    local disk_usage=$(echo "$metrics" | jq -r '.disk_usage')
    local load_avg=$(echo "$metrics" | jq -r '.load_average')
    
    # CPU alert
    if (( $(echo "$cpu_usage > 80" | bc -l) )); then
        alerts=$(echo "$alerts" | jq '. += [{"type": "cpu_high", "message": "CPU usage is high: '$cpu_usage'%", "severity": "warning"}]')
    fi
    
    # Memory alert
    if (( $(echo "$memory_usage > 85" | bc -l) )); then
        alerts=$(echo "$alerts" | jq '. += [{"type": "memory_high", "message": "Memory usage is high: '$memory_usage'%", "severity": "warning"}]')
    fi
    
    # Disk alert
    if (( $(echo "$disk_usage > 90" | bc -l) )); then
        alerts=$(echo "$alerts" | jq '. += [{"type": "disk_high", "message": "Disk usage is high: '$disk_usage'%", "severity": "critical"}]')
    fi
    
    # Load average alert
    if (( $(echo "$load_avg > 5" | bc -l) )); then
        alerts=$(echo "$alerts" | jq '. += [{"type": "load_high", "message": "System load is high: '$load_avg'", "severity": "warning"}]')
    fi
    
    echo "$alerts"
}

# Function to send alerts
send_alert() {
    local alert_type="$1"
    local message="$2"
    local severity="$3"
    
    # Log alert
    if [ "$severity" = "critical" ]; then
        error "ALERT [$alert_type]: $message"
    else
        warn "ALERT [$alert_type]: $message"
    fi
    
    # Send email alert if configured
    if command -v mail >/dev/null 2>&1 && [ -n "$ALERT_EMAIL" ]; then
        echo "Subject: Amazon AI Testing - $severity Alert" | mail -s "Amazon AI Testing - $severity Alert" "$ALERT_EMAIL"
    fi
    
    # Send Slack notification if webhook is configured
    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🚨 Amazon AI Testing Alert: $message\"}" \
            "$SLACK_WEBHOOK_URL" >/dev/null 2>&1 || true
    fi
}

# Function to save metrics
save_metrics() {
    local metrics="$1"
    local service_health="$2"
    local aws_health="$3"
    local app_health="$4"
    local alerts="$5"
    
    # Combine all metrics
    local combined_metrics=$(cat << EOF
{
    "system_metrics": $metrics,
    "service_health": $service_health,
    "aws_health": $aws_health,
    "application_health": $app_health,
    "alerts": $alerts
}
EOF
)
    
    # Save to file
    echo "$combined_metrics" > "$METRICS_FILE"
    
    # Also save to timestamped file for history
    local timestamp=$(date +%Y%m%d_%H%M%S)
    echo "$combined_metrics" > "$MONITORING_DIR/metrics_$timestamp.json"
    
    # Keep only last 1000 metric files
    ls -t "$MONITORING_DIR"/metrics_*.json | tail -n +1001 | xargs -r rm
}

# Function to generate health report
generate_health_report() {
    local metrics_file="$1"
    
    if [ ! -f "$metrics_file" ]; then
        echo "No metrics file found"
        return
    fi
    
    local system_metrics=$(cat "$metrics_file" | jq -r '.system_metrics')
    local service_health=$(cat "$metrics_file" | jq -r '.service_health')
    local alerts=$(cat "$metrics_file" | jq -r '.alerts')
    
    echo "=== Amazon AI Testing Suite Health Report ==="
    echo "Generated: $(date)"
    echo ""
    
    echo "System Metrics:"
    echo "  CPU Usage: $(echo "$system_metrics" | jq -r '.cpu_usage')%"
    echo "  Memory Usage: $(echo "$system_metrics" | jq -r '.memory_usage')%"
    echo "  Disk Usage: $(echo "$system_metrics" | jq -r '.disk_usage')%"
    echo "  Load Average: $(echo "$system_metrics" | jq -r '.load_average')"
    echo ""
    
    echo "Service Health:"
    echo "$service_health" | jq -r 'to_entries[] | "  \(.key): \(.value)"'
    echo ""
    
    echo "Active Alerts:"
    local alert_count=$(echo "$alerts" | jq 'length')
    if [ "$alert_count" -eq 0 ]; then
        echo "  No active alerts"
    else
        echo "$alerts" | jq -r '.[] | "  [\(.severity)] \(.type): \(.message)"'
    fi
    echo ""
}

# Function to monitor once
monitor_once() {
    log "Starting system monitoring cycle..."
    
    # Get system metrics
    local system_metrics=$(get_system_metrics)
    
    # Check service health
    local service_health=$(cat << EOF
{
    "docker": "$(check_service_health docker)",
    "postgres": "$(check_service_health postgres)",
    "redis": "$(check_service_health redis)",
    "prometheus": "$(check_service_health prometheus)",
    "grafana": "$(check_service_health grafana)"
}
EOF
)
    
    # Check AWS services
    local aws_health=$(check_aws_services)
    
    # Check application health
    local app_health=$(check_application_health)
    
    # Generate alerts
    local alerts=$(generate_alerts "$system_metrics")
    
    # Send alerts
    echo "$alerts" | jq -r '.[] | "\(.type) \(.message) \(.severity)"' | while read -r alert_type message severity; do
        if [ -n "$alert_type" ]; then
            send_alert "$alert_type" "$message" "$severity"
        fi
    done
    
    # Save metrics
    save_metrics "$system_metrics" "$service_health" "$aws_health" "$app_health" "$alerts"
    
    log "Monitoring cycle completed"
}

# Function to start continuous monitoring
start_monitoring() {
    log "Starting continuous system monitoring (interval: ${MONITORING_INTERVAL}s)"
    
    while true; do
        monitor_once
        sleep "$MONITORING_INTERVAL"
    done
}

# Function to show current status
show_status() {
    if [ -f "$METRICS_FILE" ]; then
        generate_health_report "$METRICS_FILE"
    else
        echo "No metrics available. Run monitoring first."
    fi
}

# Function to show help
show_help() {
    echo "Amazon AI & Robotics Testing Suite - System Monitor"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start     Start continuous monitoring"
    echo "  once      Run monitoring once"
    echo "  status    Show current system status"
    echo "  help      Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  MONITORING_INTERVAL  Monitoring interval in seconds (default: 60)"
    echo "  ALERT_EMAIL          Email address for alerts"
    echo "  SLACK_WEBHOOK_URL    Slack webhook URL for notifications"
    echo ""
}

# Main function
main() {
    # Create necessary directories
    mkdir -p "$LOGS_DIR" "$MONITORING_DIR"
    
    # Check command
    case "${1:-help}" in
        "start")
            start_monitoring
            ;;
        "once")
            monitor_once
            ;;
        "status")
            show_status
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        *)
            echo "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@" 